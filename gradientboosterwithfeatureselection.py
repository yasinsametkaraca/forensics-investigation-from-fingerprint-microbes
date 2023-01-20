import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
from sklearn.ensemble import GradientBoostingClassifier

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otuconverted.csv"

dataset = pd.read_csv(csvPath)
dataset = dataset.drop(columns=['sample'],axis=1)
dataset = dataset.drop_duplicates()

X = dataset.drop(columns=['class'])
y = dataset['class']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#this is the classifier used for feature selection
clf_feature_selection = GradientBoostingClassifier(n_estimators=50) 
rfecv = RFECV(estimator=clf_feature_selection, 
              step=1, 
              cv=5, 
              scoring = 'roc_auc',
              min_features_to_select=500
             )        

clf = GradientBoostingClassifier(n_estimators=50) 

CV_gbc = GridSearchCV(clf, 
                      param_grid={'n_estimators':[20,50,100,200], 'learning_rate': [0,10,0.25,0.5,0.75]},
                      cv= 5, scoring = 'roc_auc')

pipeline  = Pipeline([('feature_sele',rfecv),
                      ('clf_cv',CV_gbc)])

pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
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




