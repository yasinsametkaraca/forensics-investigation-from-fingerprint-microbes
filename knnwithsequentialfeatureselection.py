from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report,confusion_matrix,confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuonlyfeature.csv"
X = np.loadtxt(csvPath, delimiter=",")

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()

y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train_std, y_train)

print('Training accuracy:', np.mean(model.predict(X_train_std) == y_train)*100)
print('Test accuracy:', np.mean(model.predict(X_test_std) == y_test)*100)

sfs = SequentialFeatureSelector(model, k_features=400,forward=True,floating=False, verbose=2,scoring='accuracy',n_jobs=-1,cv=5)
sfs = sfs.fit(X_train_std, y_train)
X_train_sele = sfs.transform(X_train_std)

model.fit(X_train_sele, y_train)
X_test_sele = sfs.transform(X_test_std)

print('Training accuracy:', np.mean(model.predict(X_train_sele) == y_train)*100)
print('Test accuracy:', np.mean(model.predict(X_test_sele) == y_test)*100)

y_pred = model.predict(X_test_sele)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("AUC : ", auc(fpr, tpr))

classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :",confusion_matrix)

print("KNN Classifier Model Accuracy :", accuracy_score(y_test, y_pred))

print("Sensitivity :", confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]))
print("Specificity :", confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1]))

roc_auc = roc_auc_score(y_test, y_pred)
print("AUC ROC :", roc_auc)

scores = cross_val_score(model, X, y, cv=10)
mean_score = scores.mean()
print(mean_score)
