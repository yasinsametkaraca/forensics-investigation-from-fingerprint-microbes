from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuonlyfeature.csv"
X = np.loadtxt(csvPath, delimiter=",")

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()
y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

adaboost = AdaBoostClassifier(learning_rate=0.1,n_estimators=200)

selector = RFECV(estimator=adaboost,min_features_to_select=3302,cv=10)
selector = selector.fit(X_train, y_train)
print(selector.support_)

X_train_selected = X_train[:, selector.support_]
X_test_selected = X_test[:, selector.support_]
adaboost.fit(X_train_selected, y_train)
y_pred = adaboost.predict(X_test_selected)
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
ROC= roc_auc_score(y_test, y_pred)
print('ROC : {:.4f}'.format(ROC))

"""# Use cross-validation to evaluate the performance of the AdaBoost classifier
scores = cross_val_score(adaboost, X_train_selected, y_train, cv=10)
print("Cross-validation scores: {}".format(scores))"""
