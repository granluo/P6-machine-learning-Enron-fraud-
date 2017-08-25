#!/usr/bin/python
#!/usr/bin/env python -W ignore::DeprecationWarning

import sys
import pickle
sys.path.append(
    "../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = [
    'poi', 'deferral_payments',
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
    'director_fees',
    'to_messages',
    'from_poi_to_this_person',
    'from_messages',
    'from_this_person_to_poi',
    'shared_receipt_with_poi'
]
# You will need to use more feature


# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
pp = pprint.PrettyPrinter(indent=4)

# Task 2: Remove outliers

# The list below is to sort the people on the list by the number of NaN in features in a descending order
sorted_data_dict = sorted(data_dict.items(), key=lambda x: sum(
    1 for y in x[1] if x[1][y] == 'NaN'), reverse=True)
nan_most = sorted_data_dict[0]


# Found Lockhart Eugene has 20 features that are NaN, and the only one feature not NaN is False on poi.
# Another one is TOTAL, which represent the sum of all data and so it is not a part of the data.
data_dict.pop('TOTAL', 0)
# remove Lockhart Eugene who has 20 NaNs out of 21 features
data_dict.pop(nan_most[0], 0)
# BELFER ROBERT and BHATNAGAR SANJAY do not have correct data base of the introduction pdf and they are updated

# Update two records, which are incorrect in the dataset based on the corresponding pdf file.
data_dict['BELFER ROBERT'] = {
    'bonus': 'NaN',
    'deferral_payments': 'NaN',
    'deferred_income': -102500,
    'director_fees': 102500,
    'email_address': 'NaN',
    'exercised_stock_options': 'NaN',
    'expenses': 3285,
    'from_messages': 'NaN',
    'from_poi_to_this_person': 'NaN',
    'from_this_person_to_poi': 'NaN',
    'loan_advances': 'NaN',
    'long_term_incentive': 'NaN',
    'other': 'NaN',
    'poi': False,
    'restricted_stock': -44093,
    'restricted_stock_deferred': 44093,
    'salary': 'NaN',
    'shared_receipt_with_poi': 'NaN',
    'to_messages': 'NaN',
    'total_payments': 3285,
    'total_stock_value': 'NaN'
}

data_dict['BHATNAGAR SANJAY'] = {
    'bonus': 'NaN',
    'deferral_payments': 'NaN',
    'deferred_income': 'NaN',
    'director_fees': 'NaN',
    'email_address': 'sanjay.bhatnagar@enron.com',
    'exercised_stock_options': 15456290,
    'expenses': 137864,
    'from_messages': 29,
    'from_poi_to_this_person': 0,
    'from_this_person_to_poi': 1,
    'loan_advances': 'NaN',
    'long_term_incentive': 'NaN',
    'other': 'NaN',
    'poi': False,
    'restricted_stock': 2604490,
    'restricted_stock_deferred': -2604490,
    'salary': 'NaN',
    'shared_receipt_with_poi': 463,
    'to_messages': 523,
    'total_payments': 137864,
    'total_stock_value': 15456290
}


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import numpy as np


# From all features we found, 'from_poi_to_this_person' and 'from_this_person_to_poi' are both counting, which may be regardless of the number of mails this people send or receive.
# I will add two new features, 'ratio_from_poi' and 'ratio_to_poi', to represent the percentage of 'to_messages' and 'from_message' to poi.

def fraction_poi(data, numerator, denominator, new_feature):
    for i in data:
        data[i][new_feature] = float(data[i][numerator]) / float(
            data[i][denominator]) if data[i][denominator] != 'NaN' else 0.
    return data


# Add new key, 'ratio_from_poi', into each people
data_dict = fraction_poi(
    data_dict, 'from_poi_to_this_person', 'to_messages', 'ratio_from_poi')
# Add new key, 'percentag_to_poi', into each people
data_dict = fraction_poi(
    data_dict, 'from_this_person_to_poi', 'from_messages', 'ratio_to_poi')
# Add new features into the feature list for features split later
features_list += ['ratio_from_poi', 'ratio_to_poi']

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys=True)
print data
ALL_LABELS, ALL_FEATURES = labels, features = targetFeatureSplit(data)
print 'number of features', len(features[0])


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
train_test_split( features ,labels,
                 test_size=0.3, random_state=42)

data_train_set, _ = train_test_split( data, test_size=0.3, random_state=42)


# print my_dataset

# Check range of each feature

# this functino is to scale features hving range over 10 millions.
def feature_scaling(data_dictionary=data_dict, range_threshold=10000000, f=features):
    feature_col = zip(*f)
    features_range = []
    for i in range(len(feature_col)):
        features_range .append(
            (features_list[i + 1], max(feature_col[i]) - min(feature_col[i])))
    print sorted(features_range, key=lambda x: x[1], reverse=True)

    scale_over_10m = [features_range[x] for x in range(
        len(features_range)) if features_range[x][1] > range_threshold]

    for person in data_dictionary:
        for large_scale in scale_over_10m:
            if data_dictionary[person][large_scale[0]] != 'NaN':
                data_dictionary[person][large_scale[0]] = float(
                    data_dictionary[person][large_scale[0]]) / large_scale[1]


print '##############  Find Best Features  ##############'
print '##############  Decision Tree  ##############'
print 'features with importances larger than 0 are selected'
# Use decision tree to find importances of features


def dt_select_feature(features, labels):
    dtc = DecisionTreeClassifier()
    dtc.fit_transform(features, labels)
    dtc.feature_importances_
    list_of_features = features_list[1:]  # get rid of labels form features_list
    # a list of pairs of features and their importances
    dtc_features_importance = zip(list_of_features, dtc.feature_importances_)
# sorted the list based on the importances`
    sorted_features_importance = sorted(
    dtc_features_importance, key=lambda x: x[1], reverse=True)
    dtc_flist =  [sorted_features_importance[x][0] for x in range(
            len(sorted_features_importance)) if sorted_features_importance[x][1] > 0]
    return ({
    'selected_features':dtc_flist, # selected features from decision tree
    'features_importances':sorted_features_importance # a list of tuple containing features' names and features' importances
    })
# pp.pprint (sorted_features_importance) # print sorted importances of features of the decision tree
dtc_features = dt_select_feature(features_train, labels_train)
features_importances = dtc_features['features_importances']
dtc_feature_list = dtc_features['selected_features']
pp.pprint(features_importances)

print '##############  SelectKBest  ##############'


def skb_select_feature(all_features, all_labels, kbest = 10, flist = features_list):
    # Use SelectKBest to find kbest best features
    skb = SelectKBest(f_classif, k=kbest)
    skb.fit_transform(all_features, all_labels)
    # get the list of k all_features selected by f_classif
    skb_feature_list = np.asarray(flist[1:])[skb.get_support()]
    return ({
    'selected_features':list(skb_feature_list),# a list of selected features from SelectKBest
    'features_importances':sorted(zip(flist[1:], skb.scores_), key=lambda x: x[1], reverse=True) # a list of tuple containing features' names and features' importances
    })





print '##############  Features Selected  ##############'
# combination of two kinds of selections.
def selected_features(features,labels,sk_best):
    skb_features = skb_select_feature(features, labels, sk_best)
    dtc_features = dt_select_feature(features, labels)
    return ['poi'] + list(set(skb_features['selected_features'] + dtc_features['selected_features']))




'''
 use features selected from decition tree and select_k_best, which has k from 1
 to 20, to train the decision tree model for prediction and pick the best k that
 has the best precision and recall, both larger than 0.3, and the sum is the
 largest.
 '''
import pandas as pd
def get_k(num_iter, mds = data_train_set, feature = ALL_FEATURES, label = ALL_LABELS):
    best_k_skb = 0
    best_pr_score = 0
    record = pd.DataFrame(columns = ['k in skb','precision','recall'])
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=43)
    for i in range(0,num_iter):
        features_list = selected_features(feature,label,i)
        # pp.pprint (features_list)
        # data = featureFormat(mds, features_list, sort_keys=True)
        labels_select, features_select = targetFeatureSplit(mds)
        # labels = [int(x) for x in labels]
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features_select,labels_select,
                         test_size=0.3,random_state = 142)
        clf.fit(features_train,labels_train)
        prf = precision_recall_fscore_support(labels_test,clf.predict(features_test))

        precision = prf[0][1]
        recall = prf[1][1]
        record.loc[len(record)] = [int(i), precision, recall]
        # print precision , ' ', recall
        if (precision > 0.3) and (recall>0.3) and ((precision + recall)>best_pr_score):
            best_pr_score =precision + recall
            best_k_skb = i
    return {'best k skb': best_k_skb, 'Importance':record}

getk = get_k(20)
bestk = getk['best k skb']
importance = getk['Importance']

# print out selectkbest score
skb_features = skb_select_feature(features, labels,bestk)
features_importances = skb_features['features_importances']
skb_feature_list = skb_features['selected_features']
print 'Importance of k features selected by SelectKBest'
pp.pprint(importance)
pp.pprint(features_importances)
pp.pprint(skb_features)
print " the best k skb is ",  bestk
pp.pprint(dtc_feature_list)


features_list =selected_features(features,labels,bestk)
pp.pprint(features_list)

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict

# From all features we found, 'from_poi_to_this_person' and 'from_this_person_to_poi' are both counting, which may be regardless of the number of mails this people send or receive.
# I will add two new features, 'ratio_from_poi' and 'ratio_to_poi', to represent the percentage of 'to_messages' and 'from_message' to poi.


# Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys=True)


labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
# Set up scaled features for svm later. The scale will reduce the influence of features with big range, and a very large number may affect svm or kmeans, so I scale all of features.
scaled_features = scaling.fit_transform(features, labels)
# print scaled_features[:6]

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

##############  PCA to find variance  ##############
# from sklearn.decomposition import PCA
# pca = PCA()
# pca.fit(features)
# print pca.explained_variance_
# print len(features_list)
# print pca.explained_variance_ratio_
# pp.pprint (sorted(zip(features_list[1:],pca.explained_variance_ratio_),key = lambda x: x[1],reverse = True))


# Provided to give you a starting point. Try a variety of classifiers.

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
import pandas as pd
record = pd.DataFrame(
    columns=['algorithm', 'accuracy', 'precision', 'recall', 'f1'])

# this function is to create data frame to demonstrate accuracy, precision, recall and f score.


def record_add(recorddf, name, model, features_test, labels_test):
    predicts = model.predict(features_test)
    accuracy = accuracy_score(labels_test, predicts)
    prf = precision_recall_fscore_support(labels_test, predicts, labels=[0, 1])
    precision = prf[0][1]
    recall = prf[1][1]
    f_score = prf[2][1]
    recorddf.loc[len(recorddf)] = [name, accuracy, precision, recall, f_score]


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cross_validation import train_test_split
scaled_features_train, scaled_features_test, features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled_features, features, labels,
                     test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
clf = GaussianNB()
print "##############  Gaussian Naive Bayes  ##############"
# pip = Pipeline([('scl', MinMaxScaler()),('clf',GaussianNB())])
# pip.fit(features_train,labels_train)
# pred = pip.predict(features_test)
# print precision_recall_fscore_support(labels_test,pred)
clf.fit(features_train, labels_train)
record_add(record, 'Naive Bayes', clf, features_test, labels_test)

from sklearn import svm

clf = svm.SVC(kernel='rbf', C=2.0)
pip = Pipeline([('scl', MinMaxScaler()), ('clf', clf)])
pip.fit(features_train, labels_train)
# clf.fit(scaled_features_train,labels_train)
record_add(record, 'SVM rbf', pip, features_test, labels_test)

clf = svm.SVC(kernel='linear', C=2.0)
pip = Pipeline([('scl', MinMaxScaler()), ('clf', clf)])
pip.fit(features_train, labels_train)
record_add(record, 'SVM linear', clf, features_test, labels_test)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=43)
clf.fit(features_train, labels_train)
print 'Decision Tree'
record_add(record, 'Decision Tree', clf, features_test, labels_test)
# Create visualization of the tree
# from IPython.display import Image
# from sklearn import tree
# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file= None,
#                          feature_names=features_list[1:],
#                          class_names=features_list[0],
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
# with open('decision_tree_image12.png', 'wb') as f:
#     f.write(graph.create_png())

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=5, criterion="entropy", random_state=42)
clf.fit(features_train, labels_train)
record_add(record, 'Random Forest', clf, features_test, labels_test)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=40, learning_rate=0.5)
clf.fit(features_train, labels_train)
record_add(record, 'Ada Boost', clf, features_test, labels_test)


print "##############  Metrics of Models  ##############"
print record


np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})

from sklearn.model_selection import GridSearchCV
parameters = {'criterion': ('entropy', 'gini'), 'max_depth': [
    2, 3, 4, 5, 6, 7, 8, 9, 10]}
dtc = DecisionTreeClassifier()
clf = GridSearchCV(dtc, parameters)
clf.fit(features, labels)

print '##############  Tuning The Decision Tree  ##############'
print pd.DataFrame(
    zip(
        clf.cv_results_['param_criterion'],
        clf.cv_results_['param_max_depth'],
        clf.cv_results_['rank_test_score'],
        clf.cv_results_['split0_test_score'],
        clf.cv_results_['split0_train_score'],
        clf.cv_results_['split1_test_score'],
        clf.cv_results_['split1_train_score']),
    columns=[
        'Criterion',
        'Max depth',
        'Rank of test score',
        'split set 1 test score',
        'split set 1 train score',
        'split set 2 test score',
        'split set 2 train score'
    ]
)


# I used two criterions separtately to train the decision tree with maximum depths from 3 to 10 and found entropy generally has a hgiher accuracy than gini. and I will pick entropy for the tree's criterion.

# Since in this case, the number of POI does not take a siginificant part in the dataset. That means either the training set or test set has some probabilities to not have any POI. If the test set does not have POIs, the tree may not have a chance to test how good it is to detect POI. If the training set does not have POIs, the tree do not even has a chance to know POIs. So I will use Kfold to seperate the POIs and non-POIs dataset into training and test sets.


record_tree = pd.DataFrame(
    columns=['algorithm', 'accuracy', 'precision', 'recall', 'f1'])
set_num = 0
# from sklearn.cross_validation import KFold


# This function is to store indicies for each fold, in tuple, from kfold into list
# def kfold_sets_index(num_of_features, num_folds):
#     kf = KFold(num_of_features, num_folds)
#     sets_list = []
#     for train_indices, test_indices in kf:
#         # print('train:',train_index,'test:', test_index)
#         features_train_cv = [features_train[ii] for ii in train_indices]
#         features_test_cv = [features_train[ii] for ii in test_indices]
#         labels_train_cv = [labels_train[ii] for ii in train_indices]
#         labels_test_cv = [labels_train[ii] for ii in test_indices]
#         sets_list.append((features_train_cv, features_test_cv,
#                           labels_train_cv, labels_test_cv))
#     return sets_list

# use training set for cross_validation
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits = 4, test_size = 0.1)
def sss_sets_index(num_splits, test_set_size,features = features_train,labels = labels_train):
    sss = StratifiedShuffleSplit(n_splits = num_splits, test_size = test_set_size)

    sets_list = []
    for train_indices, test_indices in sss.split(features,labels):
        # print('train:',train_index,'test:', test_index)
        features_train_cv = [features[ii] for ii in train_indices]
        features_test_cv = [features[ii] for ii in test_indices]
        labels_train_cv = [labels[ii] for ii in train_indices]
        labels_test_cv = [labels[ii] for ii in test_indices]
        sets_list.append((features_train_cv, features_test_cv,
                          labels_train_cv, labels_test_cv))
    return sets_list



for j in (range(3, 11, 1)):
    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=j, random_state=43)
    prf_mean = []
    prf = []
    for i in sss_sets_index(4, 0.1):
        # print i
        features_train_cv, features_test_cv, labels_train_cv, labels_test_cv = i

        clf.fit(features_train_cv, labels_train_cv)
        pred = clf.predict(features_test_cv)
        prf.append(((accuracy_score(labels_test_cv, pred),) +
                    precision_recall_fscore_support(pred, labels_test_cv)))
    for col in zip(*prf):
        prf_mean.append(np.mean(col, axis=0))
    accuracy, precision, recall, f_score = prf_mean[0], prf_mean[1][1], prf_mean[2][1], prf_mean[3][1]
    record_tree.loc[len(record_tree)] = ['DT depth ' + str(j),
                                         accuracy, precision, recall, f_score]

print "##############  Average Metrics of Decision Trees with KFolds   ##############"
print record_tree


# Task 6: Dump your classifier, dataset, and features_lisprint so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your resultsself.
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=43)

dump_classifier_and_data(clf, my_dataset, features_list)
