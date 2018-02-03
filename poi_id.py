#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm, grid_search
from sklearn import cross_validation

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester

plt.style.use('ggplot')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 
                      'deferral_payments', 
                      'total_payments', 
                      'loan_advances', 
                      'bonus', 
                      'restricted_stock_deferred', 
                      'deferred_income', 
                      'total_stock_value', 
                      'expenses', 
                      'exercised_stock_options', 
                      'other', 
                      'long_term_incentive', 
                      'restricted_stock', 
                      'director_fees']

email_features = ['to_messages', 
                 'email_address', 
                 'from_poi_to_this_person', 
                 'from_messages', 
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] 

features_list = email_features + financial_features
features_list.insert(0, 'poi')
# You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Transpond dataframe
data_dict = pd.DataFrame.from_dict(data_dict)
#reorder features
data_dict = data_dict.T
data_dict = data_dict[features_list]
list(data_dict.index.values)

#Clean up
data_dict = data_dict.replace('NaN', np.nan)
data_dict = data_dict.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E'])
#remove useless (email address, director_fees restricted_stock_deferred)
del data_dict['email_address']
del data_dict['director_fees']
del data_dict['restricted_stock_deferred']

### Task 3: Create new feature(s)
data_dict['proportion_to_poi'] = data_dict['from_this_person_to_poi']/data_dict['from_messages']
data_dict['proportion_from_poi'] = data_dict['from_poi_to_this_person']/data_dict['to_messages']
data_dict = data_dict.replace('inf', 0)

features_list = features_list + ['proportion_to_poi', 'proportion_from_poi']
features_list = [e for e in features_list if e not in ('email_address', 'director_fees', 'restricted_stock_deferred')]

### Store to my_dataset for easy export below.
my_dataset = data_dict.T
my_dataset.fillna(value=0, inplace = True)

### Extract features and labels from dataset for local testing
#best_correlations_list = ['poi',
#                          'loan_advances',
#                          'exercised_stock_options',
#                          'total_stock_value',
#                          'bonus',
#                          'proportion_to_poi']

#features_list = best_correlations_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

###Lasso regression
#from sklearn.linear_model import Lasso
###features = selector.fit_transform(features, labels)
#regression = Lasso()
#regression.fit(features, labels)
#coeficients = zip(features_list[1:],regression.coef_ != 0)
#best_coefs = sorted(coeficients, key = lambda x: x[1])
##print best_coefs
#best_lasso_list = list(map(lambda x: x[0], best_coefs))
#best_lasso_list = ["poi"] + [e for e in best_lasso_list if e not in ('proportion_to_poi', 'from_poi_to_this_person', 'proportion_from_poi')]
#print("Best Lasso Regression features:")
#print best_lasso_list

#features_list = best_lasso_list


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

k=8
selector = SelectKBest(f_classif, k)
selector.fit_transform(features, labels)
#print("SelectKBest feature scores:")
scores = zip(features_list[1:],selector.scores_)
#scores = zip(features_list[1:],-np.log10(selector.pvalues_))
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
#print sorted_scores
best_kbest_list = ["poi"] + list(map(lambda x: x[0], sorted_scores))[0:k]
print("Best SelectKBest features:")
print best_kbest_list

features_list = best_kbest_list


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Feature Scaling
#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.cross_validation import KFold
#kf = KFold(len(labels), n_folds=100, shuffle = True)
#clf = GaussianNB()
#accuracy = []
#recall = []
#precision = []

#for train_indices, test_indices in kf:
#    features_train = [features[ii] for ii in train_indices]
#    features_test = [features[ii] for ii in test_indices]
#    labels_train = [labels[ii] for ii in train_indices]
#    labels_test = [labels[ii] for ii in test_indices]
#    clf.fit(features_train, labels_train)
#    pred = clf.predict(features_test)

#    accuracy.append(accuracy_score(labels_test, pred))
#    recall.append(recall_score(labels_test, pred))
#    precision.append(precision_score(labels_test, pred))

#print "accuracy: ", np.mean(accuracy)
#print "precision: ", np.mean(precision)
#print "recall: ", np.mean(recall)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#Naive Bayers
from sklearn.naive_bayes import GaussianNB
clf_n = GaussianNB()
#features_train, features_test, labels_train, labels_test = \
#train_test_split(features, labels, test_size=0.3, random_state=42)
#clf_n.fit(features_train, labels_train)
#pred = clf_n.predict(features_test)

#acc = accuracy_score(labels_test, pred)
#rec = recall_score(labels_test, pred)
#prec = precision_score(labels_test, pred)

#print "Naive bayes model"
#print "accuracy:", acc
#print "precision:", prec
#print "recall:", rec

#Support Vector Machines
#from sklearn import svm
#clf_svm = svm.SVC(kernel='linear', C=0.001, gamma=0.001)
#features_train, features_test, labels_train, labels_test = \
#train_test_split(features, labels, test_size=0.3, random_state=42)
#clf_svm.fit(features_train, labels_train)
#pred = clf_svm.predict(features_test)

#accuracy = accuracy_score(labels_test, pred)
#recall = recall_score(labels_test, pred)
#precision = precision_score(labels_test, pred)

#print "SVM"
#print "accuracy: ", accuracy
#print "precision: ", precision
#print "recall: ", recall

clf = clf_n
#clf = clf_svm


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)