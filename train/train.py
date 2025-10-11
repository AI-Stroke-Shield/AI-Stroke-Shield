import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score,f1_score, recall_score, classification_report
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression 

df = pd.read_csv('../dataset/healthcare-dataset-stroke-data.csv') 

df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

mean_bmi = df['bmi'].mean()
with open("train_results/mean_bmi.pkl", "wb") as f:
    pickle.dump(mean_bmi, f)

df2 = df.drop('id',axis=1)
df2.to_csv("train_results/dataset_dataframe.csv", index=False)
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']

oe = OrdinalEncoder()
df2[categorical_cols] = oe.fit_transform(df2[categorical_cols])

df_major = df2[(df2['stroke']==0)]
df_minor = df2[(df2['stroke']==1)]
df_minor_resmapled = resample(df_minor, replace=True, n_samples=4861, random_state=42)
df_resampled = pd.concat([df_minor_resmapled, df_major])

x = df_resampled.iloc[:,:-1]
y = df_resampled['stroke']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# --------- MODEL COMPARISON ---------

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "XGBoost": XGBClassifier(),

    
}

print("\n🔹 Model Comparison Results:")

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}\n")

# Final model training and saving
xgb = XGBClassifier()

np.save('train_results/y_test.npy', y_test)
np.save('train_results/y_train.npy', y_train)
np.save('train_results/x_train.npy', x_train)
np.save('train_results/x_test.npy', x_test)

xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)

np.save('train_results/y_pred.npy', y_pred)

print("Classification Report\n", classification_report(y_test, y_pred))
joblib.dump(xgb, '../model/xgb_boost_model.pk1')

print("✅ Model trained successfully! Classification report and encoder saved.")
