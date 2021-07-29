import numpy as np
import pandas as pd
import random,time
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics

from Measure import measure_final_score,calculate_recall,calculate_far

dataset_orig = pd.read_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\NewDebiasedDataset.csv',
                               dtype=object)
print(dataset_orig.shape)
np.random.seed(0)
## Divide into train,validation,test
dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=0,shuffle = True)

X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'action_taken'], dataset_orig_train['action_taken']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']

# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# --- CART
# clf = tree.DecisionTreeClassifier()

# clf.fit(X_train, y_train)
# import matplotlib.pyplot as plt
# y = np.arange(len(dataset_orig_train.columns)-1)
# plt.barh(y,clf.coef_[0])
# plt.yticks(y,dataset_orig_train.columns)
# plt.show()

# print(clf_male.coef_[0])
y_pred = clf.predict(X_test)
# cnf_matrix_test = confusion_matrix(y_test,y_pred)

# print(cnf_matrix_test)
# TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()




print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'accuracy'))
print("aod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
print("eod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))

print("TPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'TPR'))
print("FPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'FPR'))


# print("Precision", metrics.precision_score(y_test,y_pred))
# print("Recall", metrics.recall_score(y_test,y_pred))
# print(X_train.columns)
# print(clf.coef_)