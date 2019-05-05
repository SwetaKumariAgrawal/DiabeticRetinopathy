import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

adult_df = pd.read_csv('data.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['image', 'microaneurysmns', 'haemorrhages', 'hardexudates', 'softexudates',
                    'cottonwool','bv', 'class']
	
adult_df_rev  = adult_df
adult_df_rev.describe(include= 'all')


num_features= ['microaneurysmns', 'haemorrhages', 'hardexudates', 'softexudates',
                    'cottonwool','bv']
	
scaled_features = {}
print(adult_df_rev)
'''for each in num_features:
    mean, std = adult_df_rev[each].mean(), adult_df_rev[each].std()
    scaled_features[each] = [mean, std]
    adult_df_rev.loc[:, each] = (adult_df_rev[each] - mean)/std'''

features = adult_df_rev.values[1:,1:7]
target = adult_df_rev.values[1:,7]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.33, random_state = 10)
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)
print(features_test)
print(target_pred)
print(accuracy_score(target_test, target_pred, normalize = True))