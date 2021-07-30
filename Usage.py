import numpy as np
import pandas as pd
import random,time
import math,copy,os
from aif360.datasets import StandardDataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import accuracy_score
# from IPython.display import Markdown, display

from Measure import measure_final_score,calculate_recall,calculate_far

dataset_orig = pd.read_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\NewDebiasedDataset.csv',
                               dtype=object)
def filter_rows_by_values(df, col, values):
    return df[df[col].isin(values) == False]
dataset_orig = filter_rows_by_values(dataset_orig, "derived_race",[0.5])

print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

dataset_orig = StandardDataset(df=dataset_orig, favorable_classes=[0],
                         label_name='action_taken',
                         protected_attribute_names=['derived_race'],
                         privileged_classes=['White'])
protected_attribute_names=['derived_race']
# privileged_classes=[lambda x: x >= 25]
# features_to_drop=['personal_status', 'sex']
privileged_groups = [{'derived_race': 'White'}]
unprivileged_groups = [{'derived_race': 'Black or African American'}]


# print(dataset_orig.shape)
np.random.seed(0)
## Divide into train,validation,test

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'action_taken'], dataset_orig_train['action_taken']
# X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']

# --- LSR
# clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)
# print(prediction)
# print('Your accuracy is:', accuracy_score(y_test,prediction))
def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.statistical_parity_difference())
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


#
#
# print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'recall'))
# print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'far'))
# print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'precision'))
# print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'accuracy'))
# print("aod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'aod'))
# print("eod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'eod'))
#
# print("TPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_race', 'TPR'))
# print("FPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_race', 'FPR'))


# print("Precision", metrics.precision_score(y_test,y_pred))
# print("Recall", metrics.recall_score(y_test,y_pred))
# print(X_train.columns)
# print(clf.coef_)