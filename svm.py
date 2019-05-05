from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
# loading  dataset 
dataset = pd.read_csv("data2.csv")  
dataset.head()    
#Creating the dependent variable class
dataset.columns=['micro','hae','he','se','cw','bv','classes'] 
  
# X -> features, y -> label 
factor = pd.factorize(dataset['classes'])
dataset.classes = factor[0]
definitions = factor[1]
print(dataset.classes.head())
print(definitions)
#Splitting the data into independent and dependent variables
X = dataset.iloc[:,0:6].values
y = dataset.iloc[:,6].values
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
  
# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
print(cm)
print(accuracy)