from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import os
from datetime import datetime
from xhtml2pdf import pisa
import json
global df

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_change_me')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


models_dir = os.path.join(app.root_path, 'models')
user_data_dir = os.path.join(models_dir, 'user_data')
static_dir = os.path.join(app.root_path, 'static')
os.makedirs(static_dir, exist_ok=True)
os.makedirs(user_data_dir, exist_ok=True)

# Load model
rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))

# Base dataset (used to initialize per-user copies)
base_csv_path = os.path.join(models_dir, 'cleaned_data.csv')
base_df = pd.read_csv(base_csv_path)
if 'RecordID' not in base_df.columns:
    base_df.insert(0, 'RecordID', range(1, len(base_df) + 1))
    base_df.to_csv(base_csv_path, index=False)


def user_csv_path(user_id: int) -> str:
    return os.path.join(user_data_dir, f'cleaned_data_user_{user_id}.csv')


def ensure_user_dataset(user_id: int) -> None:
    path = user_csv_path(user_id)
    if not os.path.exists(path):
        base_df.to_csv(path, index=False)


def load_user_df(user_id: int) -> pd.DataFrame:
    ensure_user_dataset(user_id)
    dfu = pd.read_csv(user_csv_path(user_id))
    if 'RecordID' not in dfu.columns:
        dfu.insert(0, 'RecordID', range(1, len(dfu) + 1))
        dfu.to_csv(user_csv_path(user_id), index=False)
    return dfu


def save_user_df(user_id: int, dfu: pd.DataFrame) -> None:
    dfu.to_csv(user_csv_path(user_id), index=False)


def generate_plots(current_df: pd.DataFrame, user_id: int | None = None) -> dict:
    plots = {}
    suffix = f'_user_{user_id}' if user_id is not None else ''
    # Churn Distribution
    if 'Churn Value' in current_df.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x='Churn Value', data=current_df, palette='Set2')
        plt.title('Churn Distribution')
        out = os.path.join(static_dir, f'churn_plot{suffix}.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['churn_plot'] = os.path.basename(out)

    # Monthly Charges vs Churn
    if {'Monthly Charges', 'Churn Value'}.issubset(current_df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=current_df, x='Churn Value', y='Monthly Charges', palette='Set2')
        plt.title('Monthly Charges vs Churn')
        out = os.path.join(static_dir, f'monthly_charges_plot{suffix}.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['monthly_charges_plot'] = os.path.basename(out)

    # City top 10
    if 'City' in current_df.columns and 'Churn Value' in current_df.columns:
        top_cities = current_df['City'].value_counts().nlargest(10).index
        plt.figure(figsize=(10, 5))
        sns.countplot(data=current_df[current_df['City'].isin(top_cities)], x='City', hue='Churn Value', palette='coolwarm')
        plt.title('Churn by Top 10 Cities')
        plt.xticks(rotation=45)
        out = os.path.join(static_dir, f'city_plot{suffix}.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['city_plot'] = os.path.basename(out)

    # Churn by Contract Type
    if 'Contract' in current_df.columns and 'Churn Value' in current_df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=current_df, x='Contract', hue='Churn Value', palette='Set1')
        plt.title('Churn by Contract Type')
        plt.xticks(rotation=15)
        out = os.path.join(static_dir, f'contract_plot{suffix}.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['contract_plot'] = os.path.basename(out)

    # Churn by Payment Method
    if 'Payment Method' in current_df.columns and 'Churn Value' in current_df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=current_df, x='Payment Method', hue='Churn Value', palette='Set3')
        plt.title('Churn by Payment Method')
        plt.xticks(rotation=30)
        out = os.path.join(static_dir, f'payment_method_plot{suffix}.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['payment_method_plot'] = os.path.basename(out)

    # Feature Importances if shapes match
    try:
        feature_columns = [c for c in current_df.columns if c not in ['Churn Value', 'City', 'Zip Code', 'RecordID']]
        importances = rf_model.feature_importances_
        if len(importances) == len(feature_columns):
            feat_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp[:10], y=feat_imp[:10].index, palette='viridis')
            plt.title('Top 10 Feature Importances - Random Forest (Preprocessing)')
            out = os.path.join(static_dir, f'feature_importance_plot{suffix}.png')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            plots['feature_importance_plot'] = os.path.basename(out)
    except Exception:
        pass

    # --- Include additional analysis plots generated by notebook if present ---
    # These are static assets produced by the notebook; we just expose them.
    extra_files = {
        'roc_lgbm': 'roc_lgbm.png',
        'cm_lgbm': 'cm_lgbm.png',
        'roc_svc': 'roc_svc.png',
        'cm_svc': 'cm_svc.png',
        'regression_svr_true_vs_pred': 'regression_svr_true_vs_pred.png',
        'kmeans_2d': 'kmeans_2d.png',
        'kmeans_3d': 'kmeans_3d.png',
    }
    for key, fname in extra_files.items():
        path = os.path.join(static_dir, fname)
        if os.path.exists(path):
            plots[key] = os.path.basename(path)

    return plots


def predict_churn(record: dict, reference_df: pd.DataFrame) -> int:
    feature_columns = [c for c in reference_df.columns if c not in ['Churn Value', 'RecordID']]
    X = pd.DataFrame([record], columns=feature_columns)
    pred = rf_model.predict(X)[0]
    return int(pred)


def infer_schema(current_df: pd.DataFrame) -> list:
    schema = []
    for col in current_df.columns:
        if col in ['RecordID', 'Churn Value']:
            required = False
        else:
            required = True
        dtype = str(current_df[col].dtype)
        col_type = 'number' if any(x in dtype for x in ['int', 'float']) else 'text'
        # include limited choices for small-cardinality categoricals
        choices = None
        if col_type == 'text':
            unique_vals = current_df[col].dropna().unique()
            if len(unique_vals) > 0 and len(unique_vals) <= 20:
                choices = sorted([str(v) for v in unique_vals])
        schema.append({
            'name': col,
            'type': col_type,
            'required': required,
            'choices': choices
        })
    return schema


def validate_and_coerce(payload: dict, df_ref: pd.DataFrame) -> tuple[dict, list[str]]:
    errors: list[str] = []
    coerced: dict = {}
    describe = df_ref.describe(include='all')
    for col in df_ref.columns:
        if col in ['RecordID', 'Churn Value']:
            continue
        val = payload.get(col)
        if val is None:
            errors.append(f'Missing field: {col}')
            continue
        dtype = str(df_ref[col].dtype)
        if any(x in dtype for x in ['int', 'float']):
            try:
                num = float(val)
                if col in describe.columns:
                    minv = describe[col].get('min')
                    maxv = describe[col].get('max')
                    if pd.notna(minv) and num < float(minv):
                        errors.append(f'{col} below min {minv}')
                    if pd.notna(maxv) and num > float(maxv):
                        errors.append(f'{col} above max {maxv}')
                if 'int' in dtype:
                    num = int(round(num))
                coerced[col] = num
            except Exception:
                errors.append(f'{col} must be a number')
        else:
            if isinstance(val, str) and val.lower() in ['true','false']:
                coerced[col] = True if val.lower()=='true' else False
            else:
                coerced[col] = val
    return coerced, errors


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            return render_template('signup.html', error='All fields are required')
        if User.query.filter((User.username == username) | (User.email == email)).first():
            return render_template('signup.html', error='Username or email already exists')

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


# -----------------------------
# Data APIs
# -----------------------------
@app.route('/api/data', methods=['GET'])
@login_required
def get_data():
    dfu = load_user_df(current_user.id)
    return jsonify(dfu.to_dict(orient='records'))


@app.route('/api/schema', methods=['GET'])
@login_required
def get_schema():
    dfu = load_user_df(current_user.id)
    return jsonify(infer_schema(dfu))


@app.route('/api/data', methods=['POST'])
@login_required
def add_record():
    dfu = load_user_df(current_user.id)
    payload = request.get_json(force=True)
    record, errors = validate_and_coerce(payload, dfu)
    if errors:
        return jsonify({'error': 'Validation failed', 'details': errors}), 400
    new_id = int(dfu['RecordID'].max()) + 1 if not dfu.empty else 1
    record['RecordID'] = new_id
    record['Churn Value'] = predict_churn(record, dfu)
    dfu = pd.concat([dfu, pd.DataFrame([record])], ignore_index=True)
    save_user_df(current_user.id, dfu)
    generate_plots(dfu, current_user.id)
    return jsonify({'ok': True, 'record': record})


@app.route('/api/data/<int:record_id>', methods=['PUT', 'PATCH'])
@login_required
def update_record(record_id: int):
    dfu = load_user_df(current_user.id)
    payload = request.get_json(force=True)
    idx = dfu.index[dfu['RecordID'] == record_id]
    if idx.empty:
        return jsonify({'error': 'Record not found'}), 404
    i = idx[0]
    updated, errors = validate_and_coerce(payload, dfu)
    if errors:
        return jsonify({'error': 'Validation failed', 'details': errors}), 400
    for k, v in updated.items():
        if k in dfu.columns and k not in ['RecordID']:
            dfu.at[i, k] = v
    feature_columns = [c for c in dfu.columns if c not in ['Churn Value', 'RecordID']]
    record = dfu.loc[i, feature_columns].to_dict()
    dfu.at[i, 'Churn Value'] = predict_churn(record, dfu)
    save_user_df(current_user.id, dfu)
    generate_plots(dfu, current_user.id)
    return jsonify({'ok': True, 'record': dfu.loc[i].to_dict()})


@app.route('/api/data/<int:record_id>', methods=['DELETE'])
@login_required
def delete_record(record_id: int):
    dfu = load_user_df(current_user.id)
    before = len(dfu)
    dfu = dfu[dfu['RecordID'] != record_id].reset_index(drop=True)
    if len(dfu) == before:
        return jsonify({'error': 'Record not found'}), 404
    save_user_df(current_user.id, dfu)
    generate_plots(dfu, current_user.id)
    return jsonify({'ok': True})


@app.route('/api/stats', methods=['GET'])
@login_required
def stats():
    dfu = load_user_df(current_user.id)
    total = len(dfu)
    churn_rate = float(dfu['Churn Value'].mean()) if 'Churn Value' in dfu.columns and total > 0 else 0.0
    churn_counts = dfu['Churn Value'].value_counts().to_dict() if 'Churn Value' in dfu.columns else {}
    by_city = dfu['City'].value_counts().head(10).to_dict() if 'City' in dfu.columns else {}
    summary = dfu.describe(include='all').fillna('').to_dict()
    feature_columns = [c for c in dfu.columns if c not in ['Churn Value', 'City', 'Zip Code', 'RecordID']]
    importances = None
    try:
        if len(getattr(rf_model, 'feature_importances_', [])) == len(feature_columns):
            s = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
            importances = s.head(15).round(4).to_dict()
    except Exception:
        importances = None
    # Load advanced metrics saved by notebook (optional)
    metrics_path = os.path.join(models_dir, 'analysis_metrics.json')
    advanced_metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                advanced_metrics = json.load(f)
        except Exception:
            advanced_metrics = None

    return jsonify({
        'total_records': total,
        'churn_rate': churn_rate,
        'churn_counts': churn_counts,
        'top_cities': by_city,
        'summary': summary,
        'feature_importance': importances,
        'advanced_metrics': advanced_metrics
    })


@app.route('/api/analysis', methods=['GET'])
@login_required
def analysis():
    dfu = load_user_df(current_user.id)
    total = len(dfu)
    churn_rate = float(dfu['Churn Value'].mean()) if 'Churn Value' in dfu.columns and total > 0 else 0.0
    churn_counts = dfu['Churn Value'].value_counts().to_dict() if 'Churn Value' in dfu.columns else {}

    # metrics
    metrics_path = os.path.join(models_dir, 'analysis_metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    sections: list[dict] = []
    # Basic churn overview
    sections.append({
        'title': 'Churn Overview',
        'text': f"Total records: {total}. Churn rate: {churn_rate:.2%}. Counts: {churn_counts}"
    })

    # Classification analysis (LightGBM, SVC)
    if metrics and 'classification' in metrics:
        cls = metrics['classification']
        if 'lgbm' in cls:
            l = cls['lgbm']
            sections.append({
                'title': 'Classification - LightGBM',
                'text': f"Accuracy: {l.get('accuracy', 'NA')}, ROC AUC: {l.get('roc_auc', 'NA')}"
            })
        if 'svc' in cls:
            s = cls['svc']
            sections.append({
                'title': 'Classification - SVC',
                'text': f"Accuracy: {s.get('accuracy', 'NA')}, ROC AUC: {s.get('roc_auc', 'NA')}"
            })

    # Regression analysis (SVR)
    if metrics and 'regression' in metrics and 'svr_monthly_charges' in metrics['regression']:
        r = metrics['regression']['svr_monthly_charges']
        sections.append({
            'title': 'Regression - SVR (Monthly Charges)',
            'text': f"RMSE: {r.get('rmse', 'NA')}, MAE: {r.get('mae', 'NA')}, R2: {r.get('r2', 'NA')}"
        })

    # Clustering analysis (KMeans)
    if metrics and 'clustering' in metrics and 'kmeans' in metrics['clustering']:
        kinfo = metrics['clustering']['kmeans']
        sections.append({
            'title': 'Clustering - KMeans',
            'text': f"k: {kinfo.get('k', 'NA')}, features: {', '.join(kinfo.get('features_used', []))}"
        })

    return jsonify({'sections': sections})


@app.route('/api/refresh_plots', methods=['POST'])
@login_required
def refresh_plots():
    dfu = load_user_df(current_user.id)
    plots = generate_plots(dfu, current_user.id)
    urls = {k: url_for('static', filename=v) for k, v in plots.items()}
    return jsonify({'ok': True, 'plots': urls})


def render_pdf_from_html(source_html: str, output_filename: str) -> bytes:
    result = io.BytesIO()
    pisa.CreatePDF(src=source_html, dest=result)
    result.seek(0)
    return result.read()


@app.route('/report/pdf')
@login_required
def report_pdf():
    dfu = load_user_df(current_user.id)
    plots = generate_plots(dfu, current_user.id)
    # Add absolute URLs for extra plots
    plot_urls = {k: url_for('static', filename=v) for k, v in plots.items()}
    stats_resp = {
        'total_records': len(dfu),
        'churn_rate': float(dfu['Churn Value'].mean()) if 'Churn Value' in dfu.columns and len(dfu) > 0 else 0.0,
        'churn_counts': dfu['Churn Value'].value_counts().to_dict() if 'Churn Value' in dfu.columns else {},
    }
    # include feature importance if aligned
    feature_columns = [c for c in dfu.columns if c not in ['Churn Value', 'City', 'Zip Code', 'RecordID']]
    try:
        if len(getattr(rf_model, 'feature_importances_', [])) == len(feature_columns):
            s = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
            stats_resp['feature_importance'] = s.head(15).round(4).to_dict()
    except Exception:
        pass
    # Load advanced metrics saved by notebook
    metrics_path = os.path.join(models_dir, 'analysis_metrics.json')
    advanced_metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                advanced_metrics = json.load(f)
        except Exception:
            advanced_metrics = None

    html = render_template('report.html', plots=plot_urls, stats=stats_resp, advanced_metrics=advanced_metrics, now=datetime.utcnow())
    pdf_bytes = render_pdf_from_html(html, 'report.pdf')
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name='retainiq_report.pdf')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
