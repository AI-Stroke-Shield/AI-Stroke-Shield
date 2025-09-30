# 📦 Imports
import os
import pickle
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv('../dataset/healthcare-dataset-stroke-data.csv') 




df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'] = df['bmi'].fillna(df['bmi'].mean()) 


mean_bmi = df['bmi'].mean()
with open("train_results/mean_bmi.pkl", "wb") as f:
    pickle.dump(mean_bmi, f)


df2 = df.drop('id',axis=1)
df2.to_csv("train_results/dataset_dataframe.csv", index=False)
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
oe = OrdinalEncoder()
df2[categorical_cols] = oe.fit_transform(df2[categorical_cols])


df_major = df2[df2['stroke'] == 0]
df_minor = df2[df2['stroke'] == 1]
df_minor_resampled = resample(df_minor, replace=True, n_samples=4861, random_state=42)  # ✅ Fixed typo: resmapled → resampled
df_resampled = pd.concat([df_minor_resampled, df_major])


X = df_resampled.drop('stroke', axis=1)
y = df_resampled['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

np.save('train_results/y_test.npy', y_test)
np.save('train_results/y_train.npy', y_train)
np.save('train_results/x_train.npy', x_train)
np.save('train_results/x_test.npy', x_test)

xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_test)

np.save('train_results/y_pred.npy', y_pred)


print("Classification Report\n", classification_report(y_test, y_pred))


os.makedirs('../model', exist_ok=True)
joblib.dump(xgb, '../model/xgb_boost_model.pk1')
