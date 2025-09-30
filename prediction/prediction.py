import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
import pickle


df2 = pd.read_csv('../train/train_results/dataset_dataframe.csv')

loaded_model = joblib.load('../model/xgb_boost_model.pk1')
print("XGBoost model loaded successfully.")

new_user_data = {
    'gender': ['Male'],
    'age': [65],
    'hypertension': [0],
    'heart_disease': [1],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [150.0],
    'bmi': [30.0],
    'smoking_status': ['formerly smoked']
}

new_user_df = pd.DataFrame(new_user_data)

print(new_user_df)

new_user_df['bmi'] = pd.to_numeric(new_user_df['bmi'], errors='coerce')
with open("../train/train_results/mean_bmi.pkl", "rb") as f:
    mean_bmi= pickle.load(f) 
    
new_user_df['bmi'].fillna(mean_bmi, inplace=True)


# Apply ordinal encoding to categorical features
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']

combined_df_for_fitting = pd.concat([df2[categorical_cols].astype(str), new_user_df[categorical_cols].astype(str)], ignore_index=True)

oe = OrdinalEncoder()
oe.fit(combined_df_for_fitting) 


new_user_df[categorical_cols] = oe.transform(new_user_df[categorical_cols])

print(new_user_df)


predicted_class = loaded_model.predict(new_user_df)

predicted_proba = loaded_model.predict_proba(new_user_df)[:, 1]

print(f"Predicted class: {predicted_class[0]}")
print(f"Predicted probability of stroke: {predicted_proba[0]:.4f}")