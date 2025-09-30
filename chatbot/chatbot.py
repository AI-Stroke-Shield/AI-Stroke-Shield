import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
import pickle

# Load model
loaded_model = joblib.load('../model/xgb_boost_model.pk1')
st.success("XGBoost model loaded successfully.")

# Load original dataset for encoding reference
df2 = pd.read_csv('../train/train_results/dataset_dataframe.csv')

# Load mean BMI
with open("../train/train_results/mean_bmi.pkl", "rb") as f:
    mean_bmi = pickle.load(f)

st.title("Stroke Prediction App 🧠")

# Form for user input
with st.form(key='stroke_form'):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=120.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    
    submit_button = st.form_submit_button(label='Predict Stroke')

if submit_button:
    # Create new user DataFrame
    new_user_data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    }
    
    new_user_df = pd.DataFrame(new_user_data)
    
    # Fill missing BMI with mean
    new_user_df['bmi'] = pd.to_numeric(new_user_df['bmi'], errors='coerce')
    new_user_df['bmi'].fillna(mean_bmi, inplace=True)
    
    # Encode categorical features
    categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
    combined_df_for_fitting = pd.concat([df2[categorical_cols].astype(str), new_user_df[categorical_cols].astype(str)], ignore_index=True)
    oe = OrdinalEncoder()
    oe.fit(combined_df_for_fitting)
    
    new_user_df[categorical_cols] = oe.transform(new_user_df[categorical_cols])
    
    # Make prediction
    predicted_class = loaded_model.predict(new_user_df)[0]
    predicted_proba = loaded_model.predict_proba(new_user_df)[:, 1][0]
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"Predicted class: {'Stroke' if predicted_class == 1 else 'No Stroke'}")
    st.write(f"Predicted probability of stroke: {predicted_proba:.4f}")
