from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import joblib


xgb = joblib.load('../model/xgb_boost_model.pk1')
y_test = np.load('../train/train_results/y_test.npy')
x_test = np.load('../train/train_results/x_test.npy')
y_train = np.load('../train/train_results/y_train.npy')
x_train = np.load('../train/train_results/x_train.npy')
y_pred = np.load('../train/train_results/y_pred.npy')

y_test_pd = pd.DataFrame(y_test)
x_test_pd = pd.DataFrame(x_test)
y_train_pd = pd.DataFrame(y_train)
x_train_pd = pd.DataFrame(x_train)
y_pred_pd = pd.DataFrame(y_pred)




print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



y_proba = xgb.predict_proba(x_test)[:, 1]



#COnfusion matric heatmanp 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



#prediction probability
plt.hist(y_proba, bins=50, alpha=0.7)
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.grid()
plt.show()

#feature importance
from xgboost import plot_importance
plot_importance(xgb, importance_type='gain')
plt.title("Feature Importance (by Gain)")
plt.show()

#precision-recall curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()
