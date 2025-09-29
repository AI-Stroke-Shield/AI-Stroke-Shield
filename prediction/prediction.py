# Step 1: Prepare new data
# Create a DataFrame for a new user (replace with your actual data)
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

display(new_user_df)
