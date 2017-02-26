#!/usr/bin/python
### import libraries and set system path
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". 

### Create a preliminary list of features
### The final two elements in this list are new features I am creating.
### They will be explained below
features_list = ['poi', 'salary', 'total_stock_value', 'bonus', 'total_payments', \
                 'loan_advances', 'expenses', 'long_term_incentive', \
                 'exercised_stock_options']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers

### The 'TOTAL' row is an aggregation, not an individual entry. Remove it.
del data_dict['TOTAL']

### Cycle through the remaining entries and count how many missing values
### are in each entry for the features we will be examining.
for key in data_dict:
    counter = 0
    for item  in features_list:
        if data_dict[key][item] == 'NaN':
            counter = counter + 1
    print key, counter
    
### Remove the entries for which all values are missing.
del data_dict['CHAN RONNIE']
del data_dict['POWERS WILLIAM']
del data_dict['LOCKHART EUGENE E']

### Task 3: Create new feature(s)
### 

### Create 'to_poi_rate' feature.
### This tracks the percent of total emails sent that were sent to pois
### The number is stored as a ration between 0 and 1
for x in data_dict:
    if data_dict[x]['from_messages'] == 'NaN' or \
    data_dict[x]['from_this_person_to_poi'] == 'NaN':
        data_dict[x]['to_poi_rate'] = 'NaN'
    else:
        data_dict[x]['to_poi_rate'] = float(data_dict[x]['from_this_person_to_poi']) \
        / data_dict[x]['from_messages']

### Create 'from_poi_rate' feature.    
### This tracks the percent of total emails received that were sent from pois
### The number is stored as a ration between 0 and 1

for x in data_dict:
    if data_dict[x]['to_messages'] == 'NaN' or \
    data_dict[x]['from_poi_to_this_person'] == 'NaN':
        data_dict[x]['from_poi_rate'] = 'NaN'
    else:
        data_dict[x]['from_poi_rate'] = float(data_dict[x]['from_poi_to_this_person']) \
        /data_dict[x]['to_messages']

### Add both new features to the list of features we will be testing        
features_list.append('to_poi_rate')
features_list.append('from_poi_rate')  

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### import libraries for classifiers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

### create a pipeline to test data with a gaussian classifier
pipeline = Pipeline([
    ('minmax', MinMaxScaler()),
    ('kbest', SelectKBest(f_classif)),
    ('pca', PCA()),
    ('gnb', GaussianNB())
])

### import libraries with regard to validation
import sklearn.grid_search
from sklearn.cross_validation import StratifiedShuffleSplit


### set possible paramaters for the pipeline
parameters = dict(kbest__k=[3,4,5,6,7,8,9,10])

### run the pipeline with Gridsearch
sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=60)
gs = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters, cv = sss, \
                                      scoring = 'f1')
gs.fit(features, labels)

### print the optimal parameters and the corresponding score
print gs.best_params_
print gs.best_score_

### import decision tree libraries
from sklearn import tree

### create a pipeline to test date with a decision tree classifier
pipeline2 = Pipeline([
    ('minmax', MinMaxScaler()),
    ('kbest', SelectKBest(f_classif)),
    ('tree', tree.DecisionTreeClassifier())
])

### set possible parameters for the pipeline
parameters2 = dict(kbest__k=[3,4,5,6,7,8], \
                 tree__min_samples_split=[1,2,3,4,5,6,7,8,9,10,15,20,25], \
                 tree__splitter = ["best", "random"], \
#                 tree__max_depth = [None, 1, 2, 3, 4], \
                 tree__criterion = ["gini", "entropy"])

### run the pipeline with Gridsearch
gs2 = sklearn.grid_search.GridSearchCV(pipeline2, param_grid=parameters2, cv = sss, \
                                      scoring = 'f1')
gs2.fit(features, labels)

### print the optimal parameters and the corresponding score
print gs2.best_params_
print gs2.best_score_

### import Nearest Neighbors libraries
from sklearn.neighbors import KNeighborsClassifier

### create a pipeline to test data with a K Nearest Neighbors Classifier
pipeline3 = Pipeline([
    ('minmax', MinMaxScaler()),
    ('kbest', SelectKBest(f_classif)),
    ('pca', PCA()),
    ('nearest', KNeighborsClassifier())
])

### set the possible parameters for the pipeline
parameters3 = dict(kbest__k=[3,4,5,6,7,8,9,10], \
                   nearest__n_neighbors=[2,3,4,5,6,7,8,9,10], \
                   nearest__weights = ['uniform', 'distance'],
                   nearest__algorithm = ['ball_tree', 'kd_tree'])

### run the pipeline with Gridsearch
gs3 = sklearn.grid_search.GridSearchCV(pipeline3, param_grid=parameters3, cv = sss, \
                                      scoring = 'f1')
gs3.fit(features, labels)

### print the optimal parameters and the corresponding score
print gs3.best_params_
print gs3.best_score_

### Run SelectKBest to choose the optimal features
new_features = SelectKBest(f_classif, k = 7)
features_k = new_features.fit_transform(features, labels)

### Get the names of the three chosen features
feature_names = [features_list[i + 1] for i in new_features.get_support(indices=True)]
print feature_names


###Import validation & evaluation libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.cross_validation import KFold

###Use a minmax scaler for our features
features_k = MinMaxScaler().fit_transform(features_k)

###Use kfold cross validation for splitting our data into testing and training groups
kf = KFold(len(my_dataset),4, shuffle = True)
for train_indices, test_indices in kf:
    features_train = [features_k[ii] for ii in train_indices]
    features_test = [features_k[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

### Run Decision Tree Classifier and make a prediction 
    clf = tree.DecisionTreeClassifier(min_samples_split = 3, splitter = "random",\
                                     criterion = 'entropy')
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

### Print Accuracy, Recall & Precision Scores
    print accuracy_score(pred, labels_test), recall_score(labels_test, pred), \
        precision_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)