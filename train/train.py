# 📦 Imports
import os
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier

# 📄 Load dataset
df = pd.read_csv('../healthcare-dataset-stroke-data.csv')

# 🧹 Clean BMI column
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())  # ✅ Fixed chained assignment

# 💾 Save mean BMI for future use
mean_bmi = df['bmi'].mean()
with open("mean_bmi.pkl", "wb") as f:
    pickle.dump(mean_bmi, f)

# 🧠 Drop ID and encode categoricals
df2 = df.drop('id', axis=1)
df2.to_csv("dataset_dataframe.csv", index=False)

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
oe = OrdinalEncoder()
df2[categorical_cols] = oe.fit_transform(df2[categorical_cols])

# ⚖️ Resample to balance classes
df_major = df2[df2['stroke'] == 0]
df_minor = df2[df2['stroke'] == 1]
df_minor_resampled = resample(df_minor, replace=True, n_samples=4861, random_state=42)  # ✅ Fixed typo: resmapled → resampled
df_resampled = pd.concat([df_minor_resampled, df_major])

# 🎯 Split features and labels
X = df_resampled.drop('stroke', axis=1)
y = df_resampled['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 💾 Save arrays
np.save('x_train.npy', X_train)
np.save('x_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# 🧠 Train model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
np.save('y_pred.npy', y_pred)

# 📊 Print classification report
print("Classification Report\n", classification_report(y_test, y_pred))

# 💾 Save model
os.makedirs('../model', exist_ok=True)
joblib.dump(xgb, '../model/xgb_boost_model.pk1')
