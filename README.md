# RetainIQ
AI-powered e-commerce churn analysis, prediction, and reporting.

### What this project does
- Analyzes customer data and churn behavior
- Predicts churn for new or modified records
- Visualizes insights and feature importances
- Generates a PDF report with metrics and charts

## Architecture and Key Files
- `app.py`: Flask app with routes, user auth, data APIs, analysis generation, plotting, and PDF export
- `models/`:
  - `cleaned_data.csv`: preprocessed dataset used by the app (not raw `ecom.csv`)
  - `rf_model.pkl`: Random Forest classifier used for churn prediction
  - `Monthly_Charges_regressor.pkl`: optional regressor for Monthly Charges metrics (if present)
  - `kmeans_cluster.pkl`: KMeans model used for clustering metrics/visual
  - `analysis_metrics.json`: generated or notebook-produced metrics shown in the UI and PDF
- `templates/`:
  - `dashboard.html`: main UI with actions (dataset, add/modify/delete, analysis, visualize, export PDF)
  - `report.html`: PDF template embedding plots and metrics
  - `login.html`, `signup.html`
- `static/`:
  - `script.js`: client-side logic for UI, forms, data table, plot rendering, and analysis display
  - `style.css`: styles
  - plot images (generated per user), e.g. `churn_plot_user_1.png`, `feature_importance_plot_user_1.png`, `kmeans_2d_user_1.png`

## How it works (Data Flow)
1) On login, a per-user copy of the preprocessed dataset is ensured in `models/user_data/cleaned_data_user_{id}.csv`.
2) The dashboard can:
   - View dataset: loads `/api/data` and renders an editable table
   - Add/Modify/Delete records: posts to `/api/data` or `/api/data/{id}`; churn is re-predicted
   - Visualize graphs: calls `/api/refresh_plots` to generate and fetch plot URLs
   - View analysis: fetches `/api/stats` and `/api/analysis`; if missing, can POST `/api/analysis/generate`
   - Export PDF: GET `/report/pdf` renders `report.html` with metrics and images and converts to PDF

## API Endpoints (high-level)
- GET `/api/data` | POST `/api/data` | PUT/PATCH `/api/data/{id}` | DELETE `/api/data/{id}`
- GET `/api/schema` – schema for dynamic forms
- GET `/api/stats` – summary stats and top feature importances
- GET `/api/analysis` – human-readable sections + full metrics
- POST `/api/analysis/generate` – generates `analysis_metrics.json` from current user data
- GET `/api/analysis/download` – downloads metrics JSON
- POST `/api/refresh_plots` – regenerates and returns plot URLs
- GET `/report/pdf` – renders and returns a PDF report

## Visualizations generated
- Churn distribution, Monthly Charges vs Churn
- Churn by Top Cities/Contract/Payment Method
- Top 10 Feature Importances – Random Forest (horizontal bar)
- K-Means Clusters – PCA 2D scatter (with cluster counts)

## Models used
- Classification: Random Forest Classifier (`rf_model.pkl`) for churn prediction
- Regression: Monthly Charges regressor (`Monthly_Charges_regressor.pkl`) if available (used for metrics only)
- Clustering: KMeans (`kmeans_cluster.pkl`) for clustering metrics and visual (PCA scatter)

Note: The dashboard also trains a quick in-session Random Forest on the current user dataset only to render the “Top 10 Feature Importances” plot. The actual predictions use the loaded `rf_model.pkl`.

## Metrics displayed
- Classification (RF): accuracy, precision/recall/F1 (weighted), ROC AUC (if available), confusion matrix, full classification report
- Regression (Monthly Charges): RMSE, MAE, R², MSE, residual stats (and MAPE when safe)
- Clustering (KMeans): silhouette score and k

## PDF Export contents
- Overview (record counts, churn rate)
- All generated plots (embedded images)
- Detailed metrics from `analysis_metrics.json`

## Setup
1) `pip install -r requirements.txt`
2) `python app.py`
3) Open `http://127.0.0.1:5000/`

Ensure `models/cleaned_data.csv` and `models/rf_model.pkl` are present. Optional models: `kmeans_cluster.pkl`, `Monthly_Charges_regressor.pkl`.

## Notes
- The app uses the preprocessed dataset (`models/cleaned_data.csv`), not the raw `ecom.csv`.
- Metrics can be generated from the notebook into `models/analysis_metrics.json`, or via the UI button which calls `/api/analysis/generate`.
