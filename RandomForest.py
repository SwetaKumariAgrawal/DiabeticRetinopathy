import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
'''dataset = pd.read_csv("bill_authentication.csv")  
dataset.head()  
X = dataset.iloc[:, 0:4].values  
y = dataset.iloc[:, 4].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Feature Scaling
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
print(y_train)
y_pred = regressor.predict(X_test)
for i in range(len(y_pred)):
	y_pred[i]=round(y_pred[i])
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))'''
dataset = pd.read_csv("data2.csv")  
dataset.head()    
#Creating the dependent variable class
dataset.columns=['micro','hae','he','se','cw','bv','classes']
factor = pd.factorize(dataset['classes'])
dataset.classes = factor[0]
definitions = factor[1]
print(dataset.classes.head())
print(definitions)
#Splitting the data into independent and dependent variables
X = dataset.iloc[:,0:6].values
y = dataset.iloc[:,6].values
print('The independent features set: ')
print(X[:5,:])
print('The dependent variable: ')
print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
reversefactor = dict(zip(range(5),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
print(y_test)
print(y_pred)