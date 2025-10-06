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
static_dir = os.path.join(app.root_path, 'static')
os.makedirs(static_dir, exist_ok=True)

# Load model
rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))

# Load dataset and ensure a primary key exists
csv_path = os.path.join(models_dir, 'cleaned_data.csv')
df = pd.read_csv(csv_path)
if 'RecordID' not in df.columns:
    df.insert(0, 'RecordID', range(1, len(df) + 1))
    df.to_csv(csv_path, index=False)


def save_df():
    df.to_csv(csv_path, index=False)


def generate_plots(current_df: pd.DataFrame) -> dict:
    plots = {}
    # Churn Distribution
    if 'Churn Value' in current_df.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x='Churn Value', data=current_df, palette='Set2')
        plt.title('Churn Distribution')
        out = os.path.join(static_dir, 'churn_plot.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['churn_plot'] = 'churn_plot.png'

    # Monthly Charges vs Churn
    if {'Monthly Charges', 'Churn Value'}.issubset(current_df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=current_df, x='Churn Value', y='Monthly Charges', palette='Set2')
        plt.title('Monthly Charges vs Churn')
        out = os.path.join(static_dir, 'monthly_charges_plot.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['monthly_charges_plot'] = 'monthly_charges_plot.png'

    # City top 10
    if 'City' in current_df.columns and 'Churn Value' in current_df.columns:
        top_cities = current_df['City'].value_counts().nlargest(10).index
        plt.figure(figsize=(10, 5))
        sns.countplot(data=current_df[current_df['City'].isin(top_cities)], x='City', hue='Churn Value', palette='coolwarm')
        plt.title('Churn by Top 10 Cities')
        plt.xticks(rotation=45)
        out = os.path.join(static_dir, 'city_plot.png')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        plots['city_plot'] = 'city_plot.png'

    # Feature Importances if shapes match
    try:
        feature_columns = [c for c in current_df.columns if c not in ['Churn Value', 'City', 'Zip Code', 'RecordID']]
        importances = rf_model.feature_importances_
        if len(importances) == len(feature_columns):
            feat_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp[:10], y=feat_imp[:10].index, palette='viridis')
            plt.title('Top 10 Feature Importances - Random Forest')
            out = os.path.join(static_dir, 'feature_importance_plot.png')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            plots['feature_importance_plot'] = 'feature_importance_plot.png'
    except Exception:
        pass

    return plots


def predict_churn(record: dict) -> int:
    feature_columns = [c for c in df.columns if c not in ['Churn Value', 'RecordID']]
    X = pd.DataFrame([record], columns=feature_columns)
    pred = rf_model.predict(X)[0]
    return int(pred)


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
    plots = generate_plots(df)
    return render_template('dashboard.html', plots=list(plots.values()))


# -----------------------------
# Data APIs
# -----------------------------
@app.route('/api/data', methods=['GET'])
@login_required
def get_data():
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/data', methods=['POST'])
@login_required
def add_record():
    payload = request.get_json(force=True)
    required_cols = [c for c in df.columns if c not in ['RecordID', 'Churn Value']]
    for c in required_cols:
        if c not in payload:
            return jsonify({'error': f'Missing field: {c}'}), 400

    new_id = int(df['RecordID'].max()) + 1 if not df.empty else 1
    record = {c: payload.get(c) for c in required_cols}
    churn = predict_churn(record)
    record['Churn Value'] = churn
    record['RecordID'] = new_id
    global df
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save_df()
    generate_plots(df)
    return jsonify({'ok': True, 'record': record})


@app.route('/api/data/<int:record_id>', methods=['PUT', 'PATCH'])
@login_required
def update_record(record_id: int):
    payload = request.get_json(force=True)
    idx = df.index[df['RecordID'] == record_id]
    if idx.empty:
        return jsonify({'error': 'Record not found'}), 404
    i = idx[0]
    for k, v in payload.items():
        if k in df.columns and k not in ['RecordID']:
            df.at[i, k] = v

    # Recompute churn if any feature changed
    feature_columns = [c for c in df.columns if c not in ['Churn Value', 'RecordID']]
    record = df.loc[i, feature_columns].to_dict()
    df.at[i, 'Churn Value'] = predict_churn(record)
    save_df()
    generate_plots(df)
    return jsonify({'ok': True, 'record': df.loc[i].to_dict()})


@app.route('/api/data/<int:record_id>', methods=['DELETE'])
@login_required
def delete_record(record_id: int):
    global df
    before = len(df)
    df = df[df['RecordID'] != record_id].reset_index(drop=True)
    if len(df) == before:
        return jsonify({'error': 'Record not found'}), 404
    save_df()
    generate_plots(df)
    return jsonify({'ok': True})


@app.route('/api/stats', methods=['GET'])
@login_required
def stats():
    total = len(df)
    churn_rate = float(df['Churn Value'].mean()) if 'Churn Value' in df.columns and total > 0 else 0.0
    by_city = df['City'].value_counts().head(10).to_dict() if 'City' in df.columns else {}
    numeric_summary = df.describe(include='all').fillna('').to_dict()
    return jsonify({
        'total_records': total,
        'churn_rate': churn_rate,
        'top_cities': by_city,
        'summary': numeric_summary
    })


@app.route('/api/refresh_plots', methods=['POST'])
@login_required
def refresh_plots():
    plots = generate_plots(df)
    return jsonify({'ok': True, 'plots': plots})


def render_pdf_from_html(source_html: str, output_filename: str) -> bytes:
    result = io.BytesIO()
    pisa.CreatePDF(src=source_html, dest=result)
    result.seek(0)
    return result.read()


@app.route('/report/pdf')
@login_required
def report_pdf():
    plots = generate_plots(df)
    stats_resp = {
        'total_records': len(df),
        'churn_rate': float(df['Churn Value'].mean()) if 'Churn Value' in df.columns and len(df) > 0 else 0.0
    }
    html = render_template('report.html', plots=plots, stats=stats_resp, now=datetime.utcnow())
    pdf_bytes = render_pdf_from_html(html, 'report.pdf')
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name='retainiq_report.pdf')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
