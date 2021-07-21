## Here I am dividing the data first based onto protected attribute value and then train two separate models
import os
import sys
from matplotlib.pyplot import yticks, show, barh
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath('..'))
# from Measure import measure_final_score
sys.path.append(os.path.abspath('..'))
##----KEY FUNCTIONS----##
# ==========================ABOVE IMPORTS========================================
#:Training dataset D, Sensitive attribute S, Binary
# classification model M trained on D, Input space
# similarity threshold delta
def resetDataset():
    dataset_orig = pd.read_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\TestHMDA.csv', dtype=object)
    ###--------------------Sex------------------------
    indexNames1 = dataset_orig[dataset_orig['derived_sex'] == "Sex Not Available"].index
    dataset_orig.drop(indexNames1, inplace=True)
    ###-------------------Races-----------------------
    indexNames2 = dataset_orig[dataset_orig['derived_race'] == "American Indian or Alaska Native"].index
    dataset_orig.drop(indexNames2, inplace=True)
    indexNames3 = dataset_orig[dataset_orig['derived_race'] == "Native Hawaiian or Other Pacific Islander"].index
    dataset_orig.drop(indexNames3, inplace=True)
    indexNames4 = dataset_orig[dataset_orig['derived_race'] == "2 or more minority races"].index
    dataset_orig.drop(indexNames4, inplace=True)
    indexNames5 = dataset_orig[dataset_orig['derived_race'] == "Asian"].index
    dataset_orig.drop(indexNames5, inplace=True)
    indexNames6 = dataset_orig[dataset_orig['derived_race'] == "Free Form Text Only"].index
    dataset_orig.drop(indexNames6, inplace=True)
    indexNames7 = dataset_orig[dataset_orig['derived_race'] == "Race Not Available"].index
    dataset_orig.drop(indexNames7, inplace=True)
    ####----------------Ethnicity-------------------
    indexNames8 = dataset_orig[dataset_orig['derived_ethnicity'] == "Ethnicity Not Available"].index
    dataset_orig.drop(indexNames8, inplace=True)
    indexNames9 = dataset_orig[dataset_orig['derived_ethnicity'] == "Free Form Text Only"].index
    dataset_orig.drop(indexNames9, inplace=True)
    ###----------------Action_taken-----------------
    array_remove = ['4', '5', '6', '7', '8']
    def remove(array_remove):
        for beginIndex in range(len(array_remove)):
                currentIndexName = dataset_orig[dataset_orig['action_taken'] == array_remove[beginIndex]].index
                dataset_orig.drop(currentIndexName, inplace=True)
    remove(array_remove)
    dataset_orig.loc[(dataset_orig.action_taken == '1'),'action_taken'] = 0
    dataset_orig.loc[(dataset_orig.action_taken == '2'),'action_taken'] = 1
    dataset_orig.loc[(dataset_orig.action_taken == '3'),'action_taken'] = 1
    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)
    # dataset_orig.loc[(dataset_orig.derived_race == 'Joint'),'derived_race']='Joint2'
    # dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Joint'),'derived_ethnicity']='Joint1'
    ###----------------Begin Code------------------
    # print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(70))
    ####################################################################################################################################
    dataset_orig = dataset_orig.drop(['activity_year', 'lei', 'derived_msa-md', 'state_code', 'county_code', 'conforming_loan_limit', 'derived_loan_product_type', 'derived_dwelling_category', 'loan_to_value_ratio', 'interest_rate', 'rate_spread', 'total_loan_costs',    'total_points_and_fees', 'origination_charges',    'discount_points', 'lender_credits',  'loan_term',   'prepayment_penalty_term', 'intro_rate_period', 'property_value', 'multifamily_affordable_units', 'income', 'debt_to_income_ratio', 'applicant_ethnicity-2', 'applicant_ethnicity-3', 'applicant_ethnicity-4',    'applicant_ethnicity-5', 'co-applicant_ethnicity-2', 'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5', 'applicant_race-2', 'applicant_race-3', 'applicant_race-4', 'applicant_race-5', 'co-applicant_race-2',    'co-applicant_race-3', 'co-applicant_race-4', 'co-applicant_race-5', 'applicant_age_above_62', 'co-applicant_age_above_62', 'aus-2','aus-3', 'aus-4',    'aus-5', 'denial_reason-2',    'denial_reason-3', 'denial_reason-4', 'total_units', "applicant_age", "co-applicant_age"], axis=1)
    ## Change symbolics to numerics
    dataset_orig.loc[(dataset_orig.derived_sex == 'Female'),'derived_sex']= 0
    dataset_orig.loc[(dataset_orig.derived_sex == 'Male'),'derived_sex']= 1
    dataset_orig.loc[(dataset_orig.derived_sex == 'Joint'),'derived_sex']= 2
    dataset_orig.loc[(dataset_orig.derived_race == 'Black or African American'), 'derived_race'] = 0
    dataset_orig.loc[(dataset_orig.derived_race == 'White'), 'derived_race'] = 1
    dataset_orig.loc[(dataset_orig.derived_race == 'Joint'), 'derived_race'] = 2
    dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Hispanic or Latino'),'derived_ethnicity'] = 0
    dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Not Hispanic or Latino'),'derived_ethnicity'] = 1
    dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Joint'),'derived_ethnicity'] = 2

    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    # print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
    # divide the data based on sex
    # dataset_new = dataset_orig.groupby(dataset_orig['derived_sex'] == 0)
    # print(dataset_new[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
    return dataset_orig


def splittingDataset(columns, array_remove2):
    dataset_orig = resetDataset()
    for newIndex in range(len(array_remove2)):
        currentIndexName2 = dataset_orig[dataset_orig[columns] == array_remove2[newIndex]].index
        dataset_orig.drop(currentIndexName2, inplace=True)
    finalDataset = dataset_orig
    return finalDataset



def splittingDatasetSecondLayer(columns, array_remove2, initDataset1):
    initDataset = initDataset1
    for newIndex in range(len(array_remove2)):
        currentIndexName2 = initDataset[initDataset[columns] == array_remove2[newIndex]].index
        initDataset.drop(currentIndexName2, inplace=True)
    finalDataset = initDataset
    return finalDataset



allFemaleDataset = splittingDataset('derived_sex', [.5,1])
print('Printing all female \n', allFemaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

allMaleDataset = splittingDataset('derived_sex', [0,1])
print('Printing all male \n', allMaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

allJointDataset = splittingDataset('derived_sex', [0,0.5])
print('Printing all Joint \n', allJointDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


allBFdataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allFemaleDataset)
print('Printing all BFs \n', allBFdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBFdataset.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')

allBFNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allBFdataset)
print('Printing all BFNHOLs \n', allBFNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

allBFHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allBFdataset)
print('Printing all Jashan \n', allBFHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])


# divide the data based on sex
# newIndexName = dataset_orig[dataset_orig['derived_sex'] == .5].index
# dataset_orig.drop(newIndexName, inplace=True)
# newIndexName2 = dataset_orig[dataset_orig['derived_sex'] == 1].index
# dataset_orig.drop(newIndexName2, inplace=True)
# print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))

# dataset_orig = resetDataset()

# dataset_orig_male, dataset_orig_female = [x for _, x in dataset_orig.groupby(dataset_orig['derived_sex'] == 0)]
# print(dataset_orig_male[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
# print(dataset_orig_female[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))


# # divide the data based on race
# dataset_orig_male_white, dataset_orig_male_black = [x for _, x in dataset_orig_male.groupby(dataset_orig['race'] == 0)]
# dataset_orig_female_white, dataset_orig_female_black = [x for _, x in dataset_orig_female.groupby(dataset_orig['race'] == 0)]
# print(dataset_orig_male_white.shape)
# print(dataset_orig_male_black.shape)
# print(dataset_orig_female_white.shape)
# print(dataset_orig_female_black.shape)
#
# #########################################################################################################################################
# dataset_orig_male_white['race'] = 0
# dataset_orig_male_white['sex'] = 0
# X_train, y_train = dataset_orig_male_white.loc[:, dataset_orig_male_white.columns != 'Probability'], dataset_orig_male_white['Probability']
# # --- LSR
# clf1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf1.fit(X_train, y_train)
#
# # print(X_train_male['sex'])
# y = np.arange(len(dataset_orig_male_white.columns)-1)
# barh(y, clf1.coef_[0])
# yticks(y, dataset_orig_male_white.columns)
# show()
#
# print(clf1.coef_[0])
#
# #########################################################################################################################################
#
# dataset_orig_male_black['race'] = 0
# dataset_orig_male_black['sex'] = 0
# X_train, y_train = dataset_orig_male_black.loc[:, dataset_orig_male_black.columns != 'Probability'], dataset_orig_male_black['Probability']
# # --- LSR
# clf2 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf2.fit(X_train, y_train)
#
# # print(X_train_male['sex'])
# y = np.arange(len(dataset_orig_male_black.columns)-1)
# barh(y, clf2.coef_[0])
# yticks(y, dataset_orig_male_black.columns)
# show()
#
# print(clf2.coef_[0])
# #########################################################################################################################################
# dataset_orig_female_white['race'] = 0
# dataset_orig_female_white['sex'] = 0
# X_train, y_train = dataset_orig_female_white.loc[:, dataset_orig_female_white.columns != 'Probability'], dataset_orig_female_white['Probability']
# # --- LSR
# clf3 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf3.fit(X_train, y_train)
#
# # print(X_train_male['sex'])
# y = np.arange(len(dataset_orig_female_white.columns)-1)
# barh(y, clf3.coef_[0])
# yticks(y, dataset_orig_female_white.columns)
# show()
#
# print(clf3.coef_[0])
# ##########################################################################################################################################
# dataset_orig_female_black['race'] = 0
# dataset_orig_female_black['sex'] = 0
# X_train, y_train = dataset_orig_female_black.loc[:, dataset_orig_female_black.columns != 'Probability'], dataset_orig_female_black['Probability']
# # --- LSR
# clf4 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf4.fit(X_train, y_train)
#
# # print(X_train_male['sex'])
# y = np.arange(len(dataset_orig_female_black.columns)-1)
# barh(y, clf4.coef_[0])
# yticks(y, dataset_orig_female_black.columns)
# show()
#
# print(clf4.coef_[0])
#
# print(dataset_orig.shape)
# ##########################################################################################################################################
# for index, row in dataset_orig.iterrows():
#     row = [row.values[0:len(row.values) - 1]]
#     y_male_white = clf1.predict(row)
#     y_male_black = clf2.predict(row)
#     y_female_white = clf3.predict(row)
#     y_female_black = clf4.predict(row)
#     if not ((y_male_white[0] == y_male_black[0]) and (y_male_black[0] == y_female_white[0]) and (
#             y_female_white[0] == y_female_black[0])):
#         dataset_orig = dataset_orig.drop(index)
# #     else:
# #         print(y_male_white[0],y_male_black[0],y_female_white[0],y_female_black[0])
#
# print(dataset_orig.shape)
# ##########################################################################################################################################
# print(dataset_orig.shape)
# np.random.seed(0)
# ## Divide into train,validation,test
# dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=0,shuffle = True)
#
# X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
# X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']
#
# # --- LSR
# clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# # --- CART
# # clf = tree.DecisionTreeClassifier()
#
# # clf.fit(X_train, y_train)
# # import matplotlib.pyplot as plt
# # y = np.arange(len(dataset_orig_train.columns)-1)
# # plt.barh(y,clf.coef_[0])
# # plt.yticks(y,dataset_orig_train.columns)
# # plt.show()
#
# # print(clf_male.coef_[0])
# # y_pred = clf.predict(X_test)
# # cnf_matrix_test = confusion_matrix(y_test,y_pred)
#
# # print(cnf_matrix_test)
# # TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()
# #
# #
# # print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'recall'))
# # print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'far'))
# # print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'precision'))
# # print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'accuracy'))
# # print("aod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
# # print("eod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))
# #
# # print("TPR:", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'TPR'))
# # print("FPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'FPR'))
#
#
# # print("Precision", metrics.precision_score(y_test,y_pred))
# # print("Recall", metrics.recall_score(y_test,y_pred))
# # print(X_train.columns)
# # print(clf.coef_)