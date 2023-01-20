import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,accuracy_score,auc,classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()
X = dataset.drop(columns=['class'])

y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=49)

adaboost = AdaBoostClassifier(n_estimators=50)
model = adaboost.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=10)
print(scores.mean())
y_pred = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("AUC : ", auc(fpr, tpr))
classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :",confusion_matrix)
print("AdaBoost Classifier Model Accuracy :", accuracy_score(y_test, y_pred))
print("Sensitivity :", confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]))
print("Specificity :", confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1]))
roc_auc = roc_auc_score(y_test, y_pred)
print("AUC ROC :", roc_auc)

