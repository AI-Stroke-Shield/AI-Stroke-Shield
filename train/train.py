import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample

df = pd.read_csv('healthcare-dataset-stroke-data.csv') 


df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

df2 = df.drop('id',axis=1)
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']

oe = OrdinalEncoder()
df2[categorical_cols] = oe.fit_transform(df2[categorical_cols])


df_major = df2[(df2['stroke']==0)]
df_minor = df2[(df2['stroke']==1)]
df_minor_resmapled = resample(df_minor,replace=True,n_samples=4861,random_state=42)
df_resampled = pd.concat([df_minor_resmapled,df_major])

x = df_resampled.iloc[:,:-1]
y = df_resampled['stroke']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_test)
print("Classification Report", classification_report(y_test,y_pred))