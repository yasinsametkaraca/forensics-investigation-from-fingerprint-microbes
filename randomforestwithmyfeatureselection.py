import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otu.csv"                  #    Load the data into a Pandas DataFrame
dataset = pd.read_csv(csvPath)

X = dataset.iloc[1:, :].T
y = dataset.iloc[:1, :].T

classifier=RandomForestClassifier(n_estimators=120)         # I Create a random forest classifier


skf = StratifiedKFold(n_splits=5)       # Use Stratified K-Fold cross-validation to evaluate the attribute selection
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
  
    classifier.fit(X_train, y_train)              # Fit the classifier to the training data and get the feature importances
    feature_importances = classifier.feature_importances_
    
    N = 136          # Select the top N most important features
    top_features = X_train.columns[feature_importances.argsort()[::-1][:N]]          # Select the top N most important features from the test set
   
    X_test = X_test[top_features]

X_selected = X[top_features]         # Select the top N most important features from the entire dataset

model = classifier.fit(X_selected, y)           # Fit the classifier to the entire dataset using the selected features







