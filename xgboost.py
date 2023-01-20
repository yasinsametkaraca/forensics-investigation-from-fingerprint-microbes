import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import confusion_matrix, roc_auc_score,accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"

dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()

X = dataset.drop(columns=['class'])
y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7)

xgboost = xgboost.XGBClassifier()
model = xgboost.fit(X_train, y_train)
y_pred = model.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
auc_xgboost = roc_auc_score(y_test, y_pred)

print("Confusion Matrix:")
print(confusion_matrix)
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", auc_xgboost)
print("Sensitivity:", confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]))
print("Specificity:", confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1]))


