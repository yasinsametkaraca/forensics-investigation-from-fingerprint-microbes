import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix,accuracy_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder

csvPath = r"C:\Users\YSK\Documents\GitHub\pattern-recognition-with-dataset\otu.csv"
dataset = pd.read_csv(csvPath)
X = dataset.iloc[1:, :].T
y = dataset.iloc[:1, :].T
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)

model=LogisticRegression()

model.fit(X_train,y_train)
print("Test Score", model.score(X_test,y_test))
print("Train Score", model.score(X_train,y_train))


prediction=model.predict(X_test)
cm=confusion_matrix(y_test,prediction)
print("Accuracy",accuracy_score(y_test,prediction))
print('Sensitivity : ', cm[0,0]/(cm[0,0]+cm[0,1]))
print('Specificity : ', cm[1,1]/(cm[1,0]+cm[1,1]))
print("auc:",roc_auc_score(y_test,prediction))