|# Machine learning of Enron dataset analysis: detect persons-of-interest (POIs)

### Zongran Luo
### Date: 08/01/2017

## Project goal

The goal of this project is to detect persons-of-interest, POIs, based on the
Enron email dataset. This public dataset contains financial and email data
due to the Enron scandal. This dataset also include a list of persons of interest
in this fraud case.

## Dataset Overview

* The number of observations: 146
* The number of persons of interest: 18, 12% of all observations
* Features and empty rate

Features(21)  | Empty rate
--------------|------------
bonus| 0.44
    deferral_payments| 0.73
    deferred_income| 0.66
    director_fees| 0.88
    email_address| 0.24
    exercised_stock_options| 0.30
    expenses| 0.35
    from_messages| 0.41
    from_poi_to_this_person| 0.41
    from_this_person_to_poi| 0.41
    loan_advances| 0.97
    long_term_incentive| 0.55
    other| 0.36
    poi| 0.00
    restricted_stock| 0.25
    restricted_stock_deferred| 0.88
    salary| 0.35
    shared_receipt_with_poi| 0.41
    to_messages| 0.41
    total_payments| 0.14
    total_stock_value| 0.14

In this dataset, we have 20 features and one labels, poi, which is a Boolean to
identify if this person is a person of interest or not.

## Outliers and bad data

* `TOTAL` data was removed since `TOTAL` is not a person but the summary of data
 and it is an outliers
* `BELFER ROBERT` and `BHATNAGAR SANJAY` were updated with the correct data to
correspond to PDF
* `LOCKHART EUGENE E` is removed since all features are `Nan`.

## Additional features and removed features

I added some new features in the dataset.
* `ratio_from_poi`: the ratio of Emails got from POIs, ratio of `from_poi_to_this_person` to `from_messages`
* `ratio_to_poi`: the ratio of Emails sent to POIs, ratio of `from_this_person_to_poi` to `to_messages`

I add these two features because these two features can represent the percentage of Emails to or from POIs. Since simply comparing the number of emails is not fair, people sending a lot of emails to POIs which were not a big proportion may not have a higher chance than people who sent much fewer mails which, however, took a big proportion.

I also deleted some features:
* `email_address`: the Email address of person on the list.

## Features selection

I used decision tree to find importance of each feature and then features with importance not 0 will be selected.
Also I use `SelectKBest` to pick features most related to `poi`. I picked top 10 of them and combined them with features picked from the decision tree.

* Decision tree
  * I used all features and labels to train the decision tree and get the following importances.
  * Features marked as bold are picked

Features  | importances
  --------------|------------
  __exercised_stock_options__ | 0.30793650793650773
    __other__ | 0.19013605442176873
    __ratio_to_poi__ | 0.13605442176870772
    __expenses__ | 0.11934598400763817
    __shared_receipt_with_poi__ | 0.11863587540279261
    __total_stock_value__ | 0.059259259259259275
    __total_payments__ | 0.042328042328042326
    __restricted_stock__ | 0.026303854875283438
    deferral_payments| 0.0
    loan_advances| 0.0
    bonus| 0.0
    restricted_stock_deferred| 0.0
    deferred_income| 0.0
    long_term_incentive| 0.0
    director_fees| 0.0
    to_messages| 0.0
    from_poi_to_this_person| 0.0
    from_messages| 0.0
    from_this_person_to_poi| 0.0
    ratio_from_poi| 0.0)



  * SelectKBest
      * I used `SelectKBest` to pick k features, k from 1 to 20, seperately and combine with features selected from `DecisionTreeClassifier`
      * 20 combinations are applied in training a decision tree model and the k that has the best precision and recall will be picked
        * Best k followed two criterions
          * Both precision and recall are larger than 0.3
          * The sum of precision and recall is the largest among 20 all ks
        * The final k is 3        
      * I used all features and labels to run SelectKBest by computing ANOVA F-value for each feature and pick the first 3 features.
      * Features marked as bold are picked .

Features  | Score
      --------------|------------
      __total_stock_value__ | 22.782107829734311
       __exercised_stock_options__ | 22.610530706873771
       __bonus__ | 21.060001707536571
       ratio_to_poi | 16.641707070468989
       deferred_income | 11.561887713503024
       long_term_incentive | 10.072454529369441
       total_payments | 9.3802367968115874
       restricted_stock | 8.9649639563000818
       shared_receipt_with_poi | 8.7464855321290802
       loan_advances | 7.2427303965360181
       expenses| 5.5506837757329741
       from_poi_to_this_person| 5.3449415231473374
       other| 4.2198879086807812
       ratio_from_poi| 3.2107619169667441
       from_this_person_to_poi| 2.4265081272428781
       director_fees| 2.1127619890604508
       to_messages| 1.6988243485808501
       restricted_stock_deferred| 0.74349338833843037
       deferral_payments| 0.22121448377482406
       from_messages| 0.16416449823428736)

  * Selected features
    * I combined these two group of features and get the new group of features for training the model.
    * Features:
    `total_payments`,
  `loan_advances`,
  `bonus`,
  `total_stock_value`,
  `shared_receipt_with_poi`,
  `exercised_stock_options`,
  `ratio_to_poi`,
  `other`,
  `deferred_income`,
  `expenses`,
  `restricted_stock`,
  `long_term_incentive`

## Classifiers

I evaluated the following algorithms:

* Naive Bayes
* SVM with rbf kernel
* SVM linear
* Decision Tree
* Random Forest
* Ada Boost

Since this is a supervised learning and variances in features and clustering may not affect the classification most, I will not apply unsupervised learning in this project.

  algorithm |accuracy|  precision|  recall|        f1|
  ----------|--------|-----------|--------|----------|
   Naive Bayes | 0.863636 |  0.400000  |   0.4 | 0.400000
       SVM rbf | 0.886364 |  0.000000  |   0.0 | 0.000000
    SVM linear | 0.113636 |  0.113636  |   1.0 | 0.204082
 Decision Tree | 0.886364 |  0.500000  |   0.8 | 0.615385
 Random Forest | 0.909091 |  0.666667  |   0.4 | 0.500000
     Ada Boost | 0.840909 |  0.250000  |   0.2 | 0.222222

Precision, recall and f1 here and below are all for `poi` with True value.

As the table showed above, naive bayes, decision tree and random forest have satisfied the requirement of this project, both precision and recall larger than 0.3. I will pick decision tree for this project due to its high f1 score.

## Feature scaling


Since we have some financial features that can be as high as millions, I used `MinMaxScaler` to scale the features for `SVM rbf` and `SVM linear`. The large scale of the features could be one reason why SVM has lower outcome, so I scale the features to see if that can help SVM get a better result.

algorithm | accuracy | precision | recall |       f1
----------|--------|-----------|--------|----------|
  Naive Bayes | 0.863636 |  0.400000 |    0.4 | 0.400000
      SVM rbf | 0.886364 |  0.000000 |    0.0 | 0.000000
   SVM linear | 0.340909 |  0.147059 |    1.0 | 0.256410
Decision Tree | 0.886364 |  0.500000 |    0.8 | 0.615385
Random Forest | 0.886364 |  0.500000 |    0.2 | 0.285714
    Ada Boost | 0.840909 |  0.250000 |    0.2 | 0.222222

However, SVM improved but not that much. I will pick decision tree due to its high precision and recall. Though scale does not affect decision tree, I will leave the data as shown in the beginning for better interpretations.

## Metrics

Tuning the parameters means to find a combination which can bring the best predictions. I used `GridSearchCV` to create a matrix of combinations of parameters and run these combination using the model of decision tree.

Algorithms can take different parameters including:
* Criterion: including `entropy` and `gini`
* Max depth: the maximum depth of the tree, 3 to 10 in this project.

The metrics can help us avoid overfitting of the decision tree by taking a look at the accuracy of the training set and the test set. Also I will take a glance at the criterions which have entropy and gini.

Criterion | Max depth | Rank of test score | split set 1 test score  | split set 1 train score | split set 2 test score | split set 2 train score |
-----|
entropy |  3 |  1 | 0.833333 | 0.927083 | 0.895833 | 0.958333
entropy |  4 |  7 | 0.791667 | 0.937500 | 0.895833 | 0.989583
entropy |  5 |  3 | 0.812500 | 0.958333 | 0.895833 | 1.000000
entropy |  6 |  7 | 0.770833 | 0.968750 | 0.895833 | 1.000000
entropy |  7 |  2 | 0.812500 | 1.000000 | 0.895833 | 1.000000
entropy |  8 |  3 | 0.791667 | 1.000000 | 0.895833 | 1.000000
entropy |  9 |  3 | 0.812500 | 1.000000 | 0.895833 | 1.000000
entropy | 10 |  3 | 0.833333 | 1.000000 | 0.875000 | 1.000000
gini |  3 | 10 | 0.770833 | 0.947917 | 0.770833 | 0.979167
gini |  4 | 11 | 0.770833 | 0.979167 | 0.770833 | 0.989583
gini |  5 |  9 | 0.770833 | 0.989583 | 0.770833 | 1.000000
gini |  6 | 13 | 0.770833 | 1.000000 | 0.750000 | 1.000000
gini |  7 | 13 | 0.770833 | 1.000000 | 0.750000 | 1.000000
gini |  8 | 11 | 0.770833 | 1.000000 | 0.729167 | 1.000000
gini |  9 | 16 | 0.729167 | 1.000000 | 0.750000 | 1.000000
gini | 10 | 15 | 0.729167 | 1.000000 | 0.750000 | 1.000000

`Criterion`: two different types of algorithms used to calculate the impurity of the data

`Max depth`: used to limit the max depth of the tree
`Rank of test score`: the rank based on two test scores from two sets

`split set 1 test score`, `split set 1 train score`: the test score and train scores for the first split set

`split set 2 test score`, `split set 2 train score`: the test scores and test scores of the second split sets.

We can see that the train score is increasing with the increasing max depth, but the tree seems to be overfitting though not very clear in this case. In terms of the criterion, we can see the entorpy seems to have higher accuracies comparing to gini. I will pick entropy as the criterion for the model.

## Validation and cross-validation

Validation means a model trained by training set is ran on a  different data, which is a test set. If it does not work well on the test set, this model might be overfitted. New data might not work well on this model.

Based on what we have for models, 146 observations, I used cross-validation to do the verification for the decision tree model and also try to find out the best max depth for the model.

The method I used for the model is `KFold`. `KFold` is to split the dataset in to `k` parts on average. One of them will be selected as test set and the rest are going to be in the training set. Since we have `k` parts, we are going to have `k` airs of training and test sets. Since this project has only 146 observations, of which 18 are POIs. I will have the cross-validation over the test and training set.

In this case, I set `k` as 100, and so I will have 100 pairs of training and test sets. I averaged accuracy, precision, recall of 100 folds for test of each tree and I have the followings


algorithm | accuracy | precision | recall |   f1
--|
   DT depth 3  |  0.85  |   0.85 | 0.85 | 0.85
   DT depth 4  |  0.87  |   0.87 | 0.87 | 0.87
   DT depth 5  |  0.84  |   0.84 | 0.84 | 0.84
   DT depth 6  |  0.85  |   0.85 | 0.85 | 0.85
   DT depth 7  |  0.85  |   0.85 | 0.85 | 0.85
   DT depth 8  |  0.85  |   0.85 | 0.85 | 0.85
   DT depth 9  |  0.85  |   0.85 | 0.85 | 0.85
  DT depth 10  |  0.85  |   0.85 | 0.85 | 0.85

I will pick the depth with 3 to avoid overfitting.

## Outcome

The precision of the model is 0.48533	and the recall is 0.43000, both more than 0.3 and satisfying the requirements.

 -|-
--|--
Accuracy| 0.86320
Precision| 0.48533
Recall| 0.43000
F1| 0.45599
F2|0.44003
Total predictions| 15000
True positives|  860
False positives|  912
False negatives| 1140
True negatives| 12088


## Conclusion

This project is challenging due to a limited number of observations in the dataset and we only have a few POIs in it. One reason we cannot get high precision and recall is that we do not have enough samples to train the model. We also do not have enough POIs to let the model get familiar with POIs. However, if we increase the size of training set we can see the improvement, which could be found from the KFold with 100 folds. The precision and recall could be over 0.85. In this project, PCA is not considered since PCA is to find out dimensions with high variability, but we are looking for features most related to POIs, which are not necessarily connected with PCA.

## Links
https://raw.githubusercontent.com/j-bennet/udacity-nano-da/master/p5/README.md
