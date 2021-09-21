
import os
import sys

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Measure import measure_final_score

'''This file is the main file to calculate the fairness metrics of your said dataset. To do this, you want to change the dataset in line 15 to whatever
dataset you want. For instance, you will change TestHMDABalanced to your choice.'''
#Thus in binary classification, the count of true negatives is 0,0, false negatives is 1,0, true positives is 1,1, and false positives is 0,1 .
sys.path.append(os.path.abspath('..'))
fileloc = str(sys.path[0]) + '\\Data\\' + 'HMDA_2020.csv'

dataset_orig = pd.read_csv(fileloc, dtype=object)
print('Starting before split:' , dataset_orig['action_taken'].value_counts())

dataset_orig["derived_sex"] = pd.to_numeric(dataset_orig.derived_sex, errors='coerce')
dataset_orig["derived_race"] = pd.to_numeric(dataset_orig.derived_race, errors='coerce')
dataset_orig["derived_ethnicity"] = pd.to_numeric(dataset_orig.derived_ethnicity, errors='coerce')
dataset_orig["action_taken"] = pd.to_numeric(dataset_orig.action_taken, errors='coerce')


arrayDatasets = [
                'allBFNHOLdataset',
                'allBFHOLdataset',
                'allBFJointEthnicitydataset',
                'allWFNHOLdataset',
                'allWFHOLdataset',
                'allWFJointEthnicitydataset',
                'allJointRaceFemaleNHOLdataset',
                'allJointRaceFemaleHOLdataset',
                'allJointRaceFemaleJointEthnicitydataset',
                'allBMNHOLdataset',
                'allBMHOLdataset',
                'allBMJointEthnicitydataset',
                'allWMNHOLdataset',
                'allWMHOLdataset',
                'allWMJointEthnicitydataset',
                'allJointRaceMaleNHOLdataset',
                'allJointRaceMaleHOLdataset',
                'allJointRaceMaleJointEthnicitydataset',
                'allJointSexBlacksNHOLdataset',
                'allJointSexBlacksHOLdataset',
                'allJointSexBlacksJointEthnicitydataset',
                'allJointSexWhitesNHOLdataset',
                'allJointSexWhitesHOLdataset',
                'allJointSexWhitesJointEthnicitydataset',
                'allJointSexJointRaceNHOLdataset',
                'allJointSexJointRaceHOLdataset',
                'allJointSexJointRaceJointEthnicitydataset'
                 ]

sexCArray = [0,0,0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1]

raceCArray = [0,0,0,0.5,0.5,0.5,1,1,1,0,0,0,0.5,0.5,0.5,1,1,1,0,0,0,0.5,0.5,0.5,1,1,1]

ethnicityCArray = [0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1]


# print(dataset_orig.head(50))

print(dataset_orig.shape)
np.random.seed(0)
# Divide into train,validation,test

dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=0,shuffle = True)
print('train shape:', dataset_orig_train['action_taken'].value_counts())
print('test shape:', dataset_orig_test['action_taken'].value_counts())
X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'action_taken'], dataset_orig_train['action_taken']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']


# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)
# print(prediction)
# print('Your accuracy is:', accuracy_score(y_test,prediction))

# --- CART
# clf = tree.DecisionTreeClassifier()

# clf.fit(X_train, y_train)
# import matplotlib.pyplot as plt
# y = np.arange(len(dataset_orig_train.columns)-1)
# plt.barh(y,clf.coef_[0])
# plt.yticks(y,dataset_orig_train.columns)
# plt.show()

# print(clf_male.coef_[0])
# y_pred = clf.predict(X_test)
# cnf_matrix_test = confusion_matrix(y_test,y_pred)

# print(cnf_matrix_test)
# TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()



#Below is the REAL fairness metrics; you have to run them at one at a time and the only ones that currently are functional are EOD and AOD

# print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'derived_sex', 'recall'))
# print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'derived_sex', 'far'))
# print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'precision'))
# print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray,  'accuracy'))
print("mAOD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
print("mEOD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray,'mEOD'))
#
# print("Precision", metrics.precision_score(y_test,y_pred))
# print("Recall", metrics.recall_score(y_test,y_pred))
# print(X_train.columns)
# print(clf.coef_)
#





#  get_counts_editing(clf, X_train, y_train, X_test, y_test, test_df, biased_version, sexCArray, raceCArray, ethnicityCArray, metric)
#  measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_version, sexCArray, raceCArray, ethnicityCArray,  metric):












































# def describe_metrics(metrics, thresh_arr):
#     best_ind = np.argmax(metrics['bal_acc'])
#     print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
#     print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
# #     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
#     disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
#     print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
#     print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
#     print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
#     print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
#     print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
#
# metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.statistical_parity_difference())

# def filter_rows_by_values(df, col, values):
#     return df[df[col].isin(values) == False]
# dataset_orig = filter_rows_by_values(dataset_orig, "derived_race",[0.5])

#
#
# dataset_orig = StandardDataset(df=dataset_orig, favorable_classes=[0],
#                          label_name='action_taken',
#                          protected_attribute_names=['derived_race'],
#                          privileged_classes=['White'])
# protected_attribute_names=['derived_race']
# # privileged_classes=[lambda x: x >= 25]
# # features_to_drop=['personal_status', 'sex']
# privileged_groups = [{'derived_race': 'White'}]
# unprivileged_groups = [{'derived_race': 'Black or African American'}]
