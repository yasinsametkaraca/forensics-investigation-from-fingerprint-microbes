import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import MinMaxScaler

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otu.csv"
dataset = pd.read_csv(csvPath)

X = dataset.iloc[1:, :].T
X = X.drop_duplicates()

y = dataset.iloc[:1, :].T
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = GradientBoostingClassifier(n_estimators=50)
kf=KFold(n_splits=10)
score=cross_val_score(classifier,X,y,cv=kf)
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("AUC : ", auc(fpr, tpr))
classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :",confusion_matrix)
print("Gradient Boosting Classifier Model Accuracy :", accuracy_score(y_test, y_pred))
print("Sensitivity :", confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]))
print("Specificity :", confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1]))
roc_auc = roc_auc_score(y_test, y_pred)
print("AUC ROC :", roc_auc)


