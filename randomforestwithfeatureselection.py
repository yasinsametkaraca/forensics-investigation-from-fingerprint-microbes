import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"

dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()

X = dataset.drop(columns=['class'])

y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)

#this is the classifier used for feature selection
clf_feature_selection = RandomForestClassifier(n_estimators=50, 
                                        random_state=42,
                                        class_weight="balanced") 
rfecv = RFECV(estimator=clf_feature_selection, 
              step=1, 
              cv=5, 
              scoring = 'roc_auc',
              min_features_to_select=3302
             )         

clf = RandomForestClassifier(n_estimators=120, 
                             random_state=42,
                             class_weight="balanced") 
CV_rfc = GridSearchCV(clf, 
                      param_grid={'n_estimators':[20,50,100]},
                      cv= 5, scoring = 'roc_auc')

pipeline  = Pipeline([('feature_sele',rfecv),
                      ('clf_cv',CV_rfc)])

pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)

classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)

confusion_matrix = confusion_matrix(y_test, y_pred)
auc= roc_auc_score(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix)
print("Random Forest Classifier Model Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", auc)
print("Sensitivity:", confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]))
print("Specificity:", confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1]))



