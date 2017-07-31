#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
pp = pprint.PrettyPrinter(indent=4)

### Task 2: Remove outliers

# The list below is to sort the people on the list by the number of NaN in features in a descending order
sorted_data_dict = sorted(data_dict.items(), key = lambda x: sum(1 for y in x[1] if x[1][y] == 'NaN'), reverse = True)
nan_most = sorted_data_dict[0]
# pp.pprint(nan_most)


# Found Lockhart Eugene has 20 features that are NaN, and the only one feature not NaN is False on poi.
# Another one is TOTAL, which represent the sum of all data and so it is not a part of the data.
data_dict.pop('TOTAL',0)
data_dict.pop(nan_most[0],0)# remove Lockhart Eugene who has 20 NaNs out of 21 features


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()


# From all features we found, 'from_poi_to_this_person' and 'from_this_person_to_poi' are both counting, which may be regardless of the number of mails this people send or receive.
# I will add two new features, 'percentage_from_poi' and 'percentage_to_poi', to represent the percentage of 'to_messages' and 'from_message' to poi.

def fraction_poi(data,numerator,denominator,new_feature):
    for i in data:
        data[i][new_feature] = float(data[i][numerator])/float(data[i][denominator]) if data[i][denominator] != 'NaN' else 0.
        # data[i]['percentage_from_poi'] = float(data[i]['from_poi_to_this_person'])/float(data[i]['from_messages'])
    return data
data_dict = fraction_poi(data_dict,'from_poi_to_this_person','to_messages','percentage_from_poi') # Add new key, 'percentage_from_poi', into each people
data_dict = fraction_poi(data_dict,'from_this_person_to_poi','from_messages','percentage_to_poi') # Add new key, 'percentag_to_poi', into each people
features_list += ['percentage_from_poi','percentage_to_poi'] #  Add new features into the feature list for features split later

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
print data[:6]
labels, features = targetFeatureSplit(data)

# Set up scaled features for svm later. The scale will reduce the influence of features with big range, and a very large number may affect svm or kmeans, so I scale all of features.
scaled_features = scaling.fit_transform(features,labels)

# Check range of each feature
feature_col = zip(*features)

for i in range(len(feature_col)):
     print features_list[i+1],[min(feature_col[i]),max(feature_col[i])]



# pp.pprint(data_dict)
dtc = DecisionTreeClassifier()
dtc.fit_transform(features,labels)
dtc.feature_importances_
list_of_features = features_list[1:] # get rip of labels form features_list
dtc_features_importance = zip(list_of_features,dtc.feature_importances_) # a list of pairs of features and their importances
sorted_features_importance = sorted(dtc_features_importance,key = lambda x: x[1],reverse=True) # sorted the list based on the importances`
pp.pprint (sorted_features_importance) # print sorted importances of features of the decision tree
dtc_feature_list = [sorted_features_importance[x][0] for x in range(len(sorted_features_importance)) if sorted_features_importance[x][1]>0] #
pp.pprint (dtc_feature_list) # sorted importance list of features

# Use SelectKBest to find 10 best features
skb = SelectKBest(mutual_info_classif,k=10) #mutual_info_classif is to measure the dependency between variables. In this case, it is used to determine the correlations between POI and each feature.
skb.fit_transform(features,labels)
skb_feature_list = np.asarray(features_list)[skb.get_support()]# get the list of features from mutual_info_classif
pp.pprint(skb_feature_list)
print list(set(dtc_feature_list).intersection(skb_feature_list))#
print list(set(dtc_feature_list+list(skb_feature_list)))


pp.pprint(features_list)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
# From all features we found, 'from_poi_to_this_person' and 'from_this_person_to_poi' are both counting, which may be regardless of the number of mails this people send or receive.
# I will add two new features, 'percentage_from_poi' and 'percentage_to_poi', to represent the percentage of 'to_messages' and 'from_message' to poi.


### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)

# print data[:6]
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(features)
print pca.explained_variance_
print len(features_list)
print pca.explained_variance_ratio_
pp.pprint (sorted(zip(features_list[1:],pca.explained_variance_ratio_),key = lambda x: x[1],reverse = True))

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
print accuracy_score(labels_test,clf.predict(features_test))
print precision_recall_fscore_support(labels_test,clf.predict(features_test),labels = [0,1])
pp.pprint( zip(labels_test,clf.predict(features_test)))


### Task 6: Dump your classifier, dataset, and features_lisprint clf.score(clf.predict(features_test),labels_test)t so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
