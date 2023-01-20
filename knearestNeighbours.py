import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()
X = dataset.drop(columns=['class'])
y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Performing 10 fold Cross Validation
k_list = list(range(1,12))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
# K = 11 best score
classifier = KNeighborsClassifier(n_neighbors=11, metric='euclidean', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
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
