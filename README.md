 Project Overview

AI-Stroke-Shield is a machine learning–based clinical decision-support tool that predicts the likelihood of a patient having a stroke using simple health indicators such as age, BMI, glucose level, smoking status, and heart disease history.
The project leverages XGBoost for high accuracy and integrates it into a Streamlit web app for real time prediction.
This project demonstrates the use of AI in healthcare for early detection and prevention reducing hospital workload and improving timely intervention for patients at risk.

Project Workflow
1. Data Preparation

Dataset: healthcare-dataset-stroke-data.csv https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Missing values (BMI) handled using mean imputation.

Categorical data Gender, Age, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Average Glucose Level, BMI, Smoking Status.

Minority class (stroke = 1) resampled to balance the dataset.

2. Model Training

Algorithm: XGBoost (XGBClassifier)

Training/test split: 80/20

Model saved using joblib as xgb_boost_model.pk1

Evaluation metrics: Accuracy, Precision, Recall, F1-Score

3. Model Evaluation

Includes visualizations and performance metrics:

Confusion Matrix

ROC Curve

Precision–Recall Curve

Feature Importance Plot

Deployment

The trained model is deployed in a Streamlit app that allows user input through dropdowns and numeric fields.
The app predicts whether a patient is likely to experience a stroke and displays the predicted probability.

🧩 Installation Guide
1. Clone the repository
git clone https://github.com/AI-Stroke-Shield/AI-Stroke-Shield.git
cd AI-Stroke-Shield

2. Install dependencies
pip install -r requirements.txt


Typical dependencies:

streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn
joblib
pickle5

3. Run the Streamlit app
cd chatbot
streamlit run chatbot.py

How to Use

Launch the Streamlit web app.

Enter patient details:

Gender, Age, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Average Glucose Level, BMI, Smoking Status.

Click “Predict Stroke”.

The model will display:

Prediction result: Stroke / No Stroke


