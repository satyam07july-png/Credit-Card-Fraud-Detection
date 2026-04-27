# Credit-Card-Fraud-Detection
Built an end-to-end Credit Card Fraud Detection system using Machine Learning. Applied data preprocessing, scaling, and SMOTE to handle class imbalance, and trained models like Random Forest for high recall. Developed an interactive Streamlit app and FastAPI backend for real-time prediction with probability scoring.
import matplotlib.pyplot as plt
import pandas as pd

importances = model.feature_importances_

df_imp = pd.DataFrame({
    "Feature": [f"V{i}" for i in range(1,31)],
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure()
plt.barh(df_imp["Feature"][:10], df_imp["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top Features")
plt.savefig("outputs/feature_importance.png")

💳 Credit Card Fraud Detection System









🚀 Overview

An end-to-end Credit Card Fraud Detection system built using Machine Learning. This project detects fraudulent transactions by analyzing patterns in transaction data and provides real-time predictions through an interactive web application and API.

🎯 Features
🔍 Fraud Detection using ML models
⚖️ Handles class imbalance using SMOTE
⚡ Real-time prediction (Streamlit UI)
🌐 REST API using FastAPI
📊 Model evaluation (Precision, Recall, F1 Score)
🧠 Feature importance & explainability
🧠 Tech Stack
Python
Pandas, NumPy
Scikit-learn
Streamlit
FastAPI
Matplotlib, Seaborn
SHAP
📊 ML Workflow
Data Cleaning & Exploration
Feature Scaling (StandardScaler)
Handling Imbalanced Data (SMOTE)
Model Training (Logistic Regression, Random Forest)
Model Evaluation
Deployment (UI + API)
📁 Project Structure
fraud-detection/
│
├── app.py                # Streamlit UI
├── model.pkl             # Trained model
├── scaler.pkl            # Scaler
├── api/
│   └── api.py            # FastAPI backend
├── data/
├── outputs/
├── requirements.txt
└── README.md
⚡ How to Run Locally
🔹 Clone Repo
git clone https://github.com/satyam07july-png/fraud-detection.git
cd fraud-detection
🔹 Install Dependencies
pip install -r requirements.txt
🔹 Run Streamlit App
streamlit run app.py
🌐 Run API (Optional)
uvicorn api.api:app --reload

👉 Open: http://127.0.0.1:8000/docs

📈 Results
Achieved high recall for fraud detection
Reduced false negatives using SMOTE
Built a scalable ML pipeline
💼 Resume Highlight

Built an end-to-end fraud detection system using Machine Learning with real-time prediction via Streamlit and FastAPI, focusing on handling imbalanced datasets and maximizing recall.

🔥 Future Improvements
Deploy on cloud (Render / AWS)
Add authentication system
Improve UI/UX
Integrate real-time data
👨‍💻 Author

Divyansh Mishra
