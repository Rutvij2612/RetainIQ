# Step 1: Import libraries
from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # ensures plots are saved, no GUI popups
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 2: Initialize Flask app
app = Flask(__name__)

# -----------------------------
# Step 3: Load models & data
# -----------------------------
# Model path
model_path = os.path.join(app.root_path, 'models', 'rf_model.pkl')
rf_model = joblib.load(model_path)
print("Random Forest model loaded successfully!")

# Cleaned dataset path
data_path = os.path.join(app.root_path, 'models', 'cleaned_data.csv')
df = pd.read_csv(data_path)
print("Cleaned dataset loaded successfully!")

# Ensure static folder exists
static_folder = os.path.join(app.root_path, 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

# -----------------------------
# Step 4: Homepage / Login
# -----------------------------
@app.route('/')
def home():
    return render_template('login.html')


# Step 5a: Dashboard route
@app.route('/dashboard')
def dashboard():
    # Load cleaned dataset
    data_path = os.path.join('models', 'cleaned_data.csv')
    df = pd.read_csv(data_path)
    
    # List to store plot filenames
    plots = []

    # 1. Churn Distribution
    plt.figure(figsize=(5,4))
    sns.countplot(x='Churn Value', data=df, palette='Set2')
    plt.title('Churn Distribution')
    churn_plot_path = os.path.join('static', 'churn_plot.png')
    plt.savefig(churn_plot_path)
    plots.append('churn_plot.png')
    plt.close()

    # 2. Monthly Charges vs Churn
    if 'Monthly Charges' in df.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=df, x='Churn Value', y='Monthly Charges', palette='Set2')
        plt.title("Monthly Charges vs Churn")
        mc_plot_path = os.path.join('static', 'monthly_charges_plot.png')
        plt.savefig(mc_plot_path)
        plots.append('monthly_charges_plot.png')
        plt.close()

    # 3. Churn by Contract Type
    if 'Contract' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x='Contract', hue='Churn Value', palette='Set1')
        plt.title("Churn by Contract Type")
        contract_plot_path = os.path.join('static', 'contract_plot.png')
        plt.savefig(contract_plot_path)
        plots.append('contract_plot.png')
        plt.close()

    # 4. Churn by Payment Method
    if 'Payment Method' in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(data=df, x='Payment Method', hue='Churn Value', palette='Set3')
        plt.title("Churn by Payment Method")
        plt.xticks(rotation=45)
        payment_plot_path = os.path.join('static', 'payment_plot.png')
        plt.savefig(payment_plot_path)
        plots.append('payment_plot.png')
        plt.close()

    # 5. Churn by City (top 10)
    if 'City' in df.columns:
        top_cities = df['City'].value_counts().nlargest(10).index
        plt.figure(figsize=(10,5))
        sns.countplot(data=df[df['City'].isin(top_cities)], x='City', hue='Churn Value', palette='coolwarm')
        plt.title("Churn by Top 10 Cities")
        plt.xticks(rotation=45)
        city_plot_path = os.path.join('static', 'city_plot.png')
        plt.savefig(city_plot_path)
        plots.append('city_plot.png')
        plt.close()

    # 6. Feature Importance (Random Forest)
    try:
        importances = rf_model.feature_importances_
        features = df.drop(columns=['Churn Value', 'City', 'Zip Code'], errors='ignore').columns
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x=feat_imp[:10], y=feat_imp[:10].index, palette='viridis')
        plt.title("Top 10 Feature Importances - Random Forest")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        feat_plot_path = os.path.join('static', 'feature_importance_plot.png')
        plt.savefig(feat_plot_path)
        plots.append('feature_importance_plot.png')
        plt.close()
    except Exception as e:
        print("⚠️ Could not generate feature importance plot:", e)

    # Render dashboard with all plots
    return render_template('dashboard.html', plots=plots)


# -----------------------------
# Step 6: Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
