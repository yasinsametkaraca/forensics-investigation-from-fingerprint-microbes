import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train.values, y_train)
scores = cross_val_score(classifier, X_train.values, y_train, cv=5)
print("Cross-validation scores: mean = {:.2f}, std = {:.2f}".format(scores.mean(), scores.std()))
y_pred = classifier.predict(X_test)

cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
print("Test values that were inputted:\n")
print(X_test.values)
print("Predictions for those corresponding inputs:\n")
predictions = classifier.predict(X_test.values)
print(predictions)
print("How accurate these predictions were:\n")
score = accuracy_score(y_test,predictions)
print("Accuracy: ",score)
sens = cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1])
print('Sensitivity : ', sens)
spec = cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1])
print('Specificity : ', spec)
print("AUC: {:.2f}".format(scores.mean()))



