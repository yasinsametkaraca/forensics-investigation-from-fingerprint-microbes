import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,auc
from sklearn.metrics import roc_auc_score

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otu.csv"

dataset = pd.read_csv(csvPath)
X = dataset.iloc[1:, :].T

y = dataset.iloc[:1, :].T
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 0)

classifier = MLPClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)

classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy, "\n")

sensitivity = cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1])
print('Sensitivity : ', sensitivity)

specificity = cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1])
print('Specificity : ', specificity)

ROC_AUC = roc_auc_score(y_test, y_pred)
print('ROC AUC : {:.4f}'.format(ROC_AUC))