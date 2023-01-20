import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"

dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()

X = dataset.drop(columns=['class'])
y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Split the data into training and test sets

parameters = {'n_estimators': [10, 50, 100, 200],  # Set up the parameters for the grid search
              'learning_rate': [0.1, 0.5, 1.0, 2.0]}

adaboost = AdaBoostClassifier() # Create the AdaBoost classifier
grid_search = GridSearchCV(adaboost, parameters, cv=5) # Create the grid search object

grid_search.fit(X_train, y_train)  # Fit the grid search object to the training data

print("Best parameters:", grid_search.best_params_) # Print the best parameters

adaboost = AdaBoostClassifier(**grid_search.best_params_) # Use the best parameters to create the final AdaBoost classifier
adaboost.fit(X_train, y_train)
predictions = adaboost.predict(X_test) # Make predictions on the test data
 

accuracy = accuracy_score(y_test, predictions)  # Calculate the accuracy of the predictions
print("Accuracy:", accuracy)

cMatrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cMatrix)

classificationReport = classification_report(y_test, predictions)
print("Classification Report:\n", classificationReport)

sensitivity = cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1])
print('Sensitivity : ', sensitivity)

specificity = cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1])
print('Specificity : ', specificity)

auc_adaboost = roc_auc_score(y_test, predictions)
print("AUC:", auc_adaboost)