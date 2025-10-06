# RetainIQ
🧠 AI-Powered E-Commerce Customer Behavioral Analysis and Churn Prediction
Predict. Retain. Grow.

A Machine Learning-driven web application for E-commerce customer behavior analysis, churn prediction, and interactive data visualization.

🚀 Overview

RetainIQ is an end-to-end intelligent analytics system that helps businesses:

Analyze customer data

Predict churn likelihood

Visualize key insights in an interactive dashboard

Add, update, or delete records dynamically

Generate downloadable PDF reports

It combines Flask, Machine Learning, and Data Visualization into one unified platform.

🧩 Key Features

✅ User Authentication
Secure Login & Signup pages inspired by modern dashboards.

✅ Interactive Dashboard
Explore churn distribution, customer contracts, and payment behavior visually.

✅ Smart Predictions
Input new customer data and instantly predict churn probability.

✅ Editable Data Table
Modify, delete, or add new entries — your visualizations auto-update.

✅ Downloadable Reports
Export analysis and visualizations as a professionally styled PDF report.

✅ Model Integration
Powered by a trained Random Forest Classifier built on cleaned and preprocessed data.

🧠 Tech Stack
Layer	Technology
Frontend	HTML, CSS, Bootstrap, JavaScript
Backend	Flask (Python)
ML Model	Scikit-learn (Random Forest, Logistic Regression)
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Storage	CSV / Local SQLite (optional)
Export	PDFKit / ReportLab
⚙️ Project Structure
RetainIQ/
│
├── app.py                  # Flask application
├── requirements.txt        # Dependencies
├── .gitignore
├── models/
│   ├── rf_model.pkl
│   └── cleaned_data.csv
│
├── templates/
│   ├── login.html
│   ├── signup.html
│   └── dashboard.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── plots/
│
├── RetainIQ_Analysis.ipynb # Jupyter Notebook (data cleaning + model training)
└── README.md

⚡ Getting Started
1️⃣ Clone the Repository
git clone https://github.com/Rutvij2612/RetainIQ.git
cd RetainIQ

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
python app.py


Then open: http://127.0.0.1:5000/

📊 Machine Learning Workflow

Data cleaning and preprocessing (RetainIQ_Analysis.ipynb)

Feature encoding and scaling

Model training (Random Forest + Logistic Regression)

Evaluation using accuracy, precision, recall, and F1-score

Exporting final model (rf_model.pkl) for app integration

🧾 Future Enhancements

Integration with live database (PostgreSQL / Firebase)

Email alert system for churn predictions

Admin analytics panel

Multi-user dashboard with role management

🤝 Contribution

Fork this repository

Create your feature branch

Commit your changes

Push and open a Pull Request

🧑‍💻 Author

Rutvij @Rutvij2612

📧 AI, ML & Full-Stack Developer | CSE (B.Tech) 5th Sem
