import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

kaynak = "https://www.naukri.com/learning/articles/random-forest-algorithm-python-code/"
csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"

dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()

X = dataset.drop(columns=['class'])
y = dataset['class']

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)   

classifier=RandomForestClassifier(n_estimators=120)
model = classifier.fit(X_train, y_train)
y_pred = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("AUC : ", auc(fpr, tpr))

classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :",confusion_matrix)
print("Gradient Booster Classifier Model Accuracy :", accuracy_score(y_test, y_pred))
print("Sensitivity :", confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]))
print("Specificity :", confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1]))
roc_auc = roc_auc_score(y_test, y_pred)
print("AUC ROC :", roc_auc)

scores = cross_val_score(classifier, X, y, cv=10)  #I Use cross-validation to evaluate the performance of the AdaBoost classifier
print("Cross-validation scores: {}".format(scores))
print("Mean score: {:.2f}%".format(scores.mean() * 100))
