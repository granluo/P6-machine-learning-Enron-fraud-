# Machine-learning-Enron-fraud
## Project goal

The goal of this project is to detect persons-of-interest, POIs, based on the
Enron email dataset. This public dataset contains financial and email data
due to the Enron scandal. This dataset also include a list of persons of interest
in this fraud case. You are free to check out the project report (right here)[https://github.com/ollkorrect/P6-machine-learning-Enron-fraud-/blob/master/Report.md]

## Why machine learning?

I used machine learning to deal with this problem, since, in this dataset, we do not have a large size of sample, and the number of person of interest is very small. There is no one or two features that are very related to POIs and in this case, we cannot simply use linear regression with one or two variables. POIs could be related to many factors and these factors may or may not be correlated, so the model for predicting POIs could be complex. Machine learning can refine the model based on datasets we have. Machine learning, hence, is very necessary to be applied in this problem. In this project, I will try to find a best-fit model for prediction by cleaning outliers, selecting features, selecting models and tuning parameters.

## Dataset Overview

* The number of observations: 146
* The number of persons of interest: 18, 12% of all observations
* Features and empty rate

In this dataset, we have 20 features and one labels, poi, which is a Boolean to
identify if this person is a person of interest or not.

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

## Classifiers

I evaluated the following algorithms:

* Naive Bayes
* SVM with rbf kernel
* SVM linear
* Decision Tree
* Random Forest
* Ada Boost

## Validation and cross-validation

Validation means a model trained by training set is ran on a | different data, which is a test set. If it does not work well on the test set, this model might be overfitted. New data might not work well on this model.

Based on what we have for models, 146 observations, I used cross-validation to do the verification for the decision tree model and also try to find out the best max depth for the model.

The method I used for the model is `StratifiedShuffleSplit`. `StratifiedShuffleSplit` is to shuffle the dataset in to `n` times. For each time, a certain proportion of the data will be selected to be test set and the rest are training set.  Since this project has only 146 observations, of which 18 are POIs. I will have the cross-validation over the test and training set.


## Outcome

The precision of the model was 0.53477, which meant among all pois them model
predicts, 53.5 percent of them were correct. And the recall was 0.48450, so 48.5
percent of pois were correctely founded among all POIs. These two criterions were
both more than 0.3 and satisfying the requirements.

 -|-
--|--
Accuracy| 0.87507
Precision| 0.53477
Recall| 0.48450
F1| 0.50839
F2|0.49378
Total predictions| 15000
True positives|  969
False positives|  843
False negatives| 1031
True negatives| 12157


## Conclusion

This project is challenging due to a limited number of observations in the dataset and we only have a few POIs in it. One reason we cannot get high precision and recall is that we do not have enough samples to train the model. We also do not have enough POIs to let the model get familiar with POIs. However, if we increase the size of training set we can see the improvement, which could be found from the KFold with 100 folds. The precision and recall could be over 0.85. In this project, PCA is not considered since PCA is to find out dimensions with high variability, but we are looking for features most related to POIs, which are not necessarily connected with PCA.
