import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
import pickle

# Step 1: Prepare new data
# Create a DataFrame for a new user (replace with your actual data)
df2 = pd.read_csv('../train/dataset_dataframe.csv')

loaded_model = joblib.load('../model/xgb_boost_model.pk1')
print("XGBoost model loaded successfully.")

new_user_data = {
    'gender': ['Male'],
    'age': [65.0],
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

# Step 2: Load the model

# Step 3: Preprocess the new data
# Handle missing BMI values (using the mean from the training data, if available, or recalculate)
# Since we don't have the original mean readily available, we'll use the mean from the existing df for demonstration
# In a real scenario, you would save and load the mean from the training data
new_user_df['bmi'] = pd.to_numeric(new_user_df['bmi'], errors='coerce')
with open("../train/mean_bmi.pkl", "rb") as f:
    mean_bmi= pickle.load(f) 
    
new_user_df['bmi'].fillna(mean_bmi, inplace=True)


# Apply ordinal encoding to categorical features
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
# We need to use the same OrdinalEncoder fitted on the training data
# To handle potential new categories in new_user_df, we'll refit the encoder
# on a combined dataset (original + new user data) for demonstration.
# In a real scenario, you would save and load the fitted encoder and potentially
# use handle_unknown='use_encoded_value' or fit on a larger, representative dataset.

# Combine original and new data for fitting the encoder
combined_df_for_fitting = pd.concat([df2[categorical_cols].astype(str), new_user_df[categorical_cols].astype(str)], ignore_index=True)

oe = OrdinalEncoder()
oe.fit(combined_df_for_fitting) # Fit on the combined data

# Transform the new user data
new_user_df[categorical_cols] = oe.transform(new_user_df[categorical_cols])

print(new_user_df)


# Step 4: Predict stroke
# Predict the class (0 or 1)
predicted_class = loaded_model.predict(new_user_df)

# Predict the probability of stroke
predicted_proba = loaded_model.predict_proba(new_user_df)[:, 1]

print(f"Predicted class: {predicted_class[0]}")
print(f"Predicted probability of stroke: {predicted_proba[0]:.4f}")