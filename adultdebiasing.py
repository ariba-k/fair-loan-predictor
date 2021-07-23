## Here I am dividing the data first based onto protected attribute value and then train two separate models
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath('..'))
# from Measure import measure_final_score
sys.path.append(os.path.abspath('..'))
##----KEY FUNCTIONS----##
## Load dataset
def resetDataset():
    dataset_orig = pd.read_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\adult.csv')
    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()
    ## Drop categorical features
    dataset_orig = dataset_orig.drop( ['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'], axis=1)
    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)
    # print(dataset_orig.head(40))
    ## Discretize age
    dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 60) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 50) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 40) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 30) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 20) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 10) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
    dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])

    # print((dataset_orig['sex'] == 0).head(38))
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    # print(dataset_orig[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(38))
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



allFemaleDataset = splittingDataset('sex', [1])
# print('Printing all female \n', allFemaleDataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

def allFemaleReset():
    allFemaleDataset = splittingDataset('sex', [1])
    return allFemaleDataset

allMaleDataset = splittingDataset('sex', [0])
# print('Printing all male \n', allMaleDataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

def allMaleReset():
    allMaleDataset = splittingDataset('sex', [0])
    return allMaleDataset

allBFdataset = splittingDatasetSecondLayer('race', [1], allFemaleDataset)
# print('Printing all Black-Females \n', allBFdataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

allFemaleDataset = allFemaleReset()

allWFdataset = splittingDatasetSecondLayer('race', [0], allFemaleDataset)
# print('Printing all White-Females \n', allWFdataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

allBMdataset = splittingDatasetSecondLayer('race', [1], allMaleDataset)
# print('Printing all Black-Males \n', allBMdataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

allMaleDataset = allMaleReset()

allWMdataset = splittingDatasetSecondLayer('race', [0], allMaleDataset)
# print('Printing all White-Males \n', allWMdataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

# # divide the data based on sex

# dataset_orig = resetDataset()
#
# dataset_orig_male, dataset_orig_female = [x for _, x in dataset_orig.groupby(dataset_orig['sex'] == 0)]
# print(dataset_orig_male[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(20))
# print(dataset_orig_female[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(20))
#

#
# # divide the data based on race
# dataset_orig_male_white, dataset_orig_male_black = [x for _, x in dataset_orig_male.groupby(dataset_orig['race'] == 0)]
# dataset_orig_female_white, dataset_orig_female_black = [x for _, x in dataset_orig_female.groupby(dataset_orig['race'] == 0)]


def createClassifier(D):
    D['race'] = 0
    D['sex'] = 0

    X_train, y_train = D.loc[:, D.columns != 'Probability'], D['Probability']
    # --- LSR
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    return clf.fit(X_train, y_train)

clf1 = createClassifier(allWMdataset)
clf2 = createClassifier(allBMdataset)
clf3 = createClassifier(allWFdataset)
clf4 = createClassifier(allBFdataset)
#--------------------Classifiers---------------------
# allWMdataset['race'] = 0
# allWMdataset['sex'] = 0
#
# X_train, y_train = allWMdataset.loc[:, allWMdataset.columns != 'Probability'], allWMdataset['Probability']
# # --- LSR
# clf1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf1.fit(X_train, y_train)
#
# # print(X_train_male['sex'])
# import matplotlib.pyplot as plt
# # y = np.arange(len(allWMdataset.columns)-1)
# # plt.barh(y,clf1.coef_[0])
# # plt.yticks(y,allWMdataset.columns)
# # plt.show()
# #
# # print(clf1.coef_[0])
#
#
# #------------------------------------------------------------------------------------
# allBMdataset['race'] = 0
# allBMdataset['sex'] = 0
# X_train, y_train = allBMdataset.loc[:, allBMdataset.columns != 'Probability'], allBMdataset['Probability']
# # --- LSR
# clf2 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf2.fit(X_train, y_train)
#
# # # print(X_train_male['sex'])
# # import matplotlib.pyplot as plt
# # y = np.arange(len(allBMdataset.columns)-1)
# # plt.barh(y,clf2.coef_[0])
# # plt.yticks(y,allBMdataset.columns)
# # plt.show()
#
# # print(clf2.coef_[0])
#
# #------------------------------------------------------------------------------------
# allWFdataset['race'] = 0
# allWFdataset['sex'] = 0
# X_train, y_train = allWFdataset.loc[:, allWFdataset.columns != 'Probability'], allWFdataset['Probability']
# # --- LSR
# clf3 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf3.fit(X_train, y_train)
#
# # # print(X_train_male['sex'])
# # import matplotlib.pyplot as plt
# # y = np.arange(len(allWFdataset.columns)-1)
# # plt.barh(y,clf3.coef_[0])
# # plt.yticks(y,allWFdataset.columns)
# # plt.show()
#
# # print(clf3.coef_[0])
#
# #------------------------------------------------------------------------------------
# allBFdataset['race'] = 0
# allBFdataset['sex'] = 0
# X_train, y_train = allBFdataset.loc[:, allBFdataset.columns != 'Probability'], allBFdataset['Probability']
# # --- LSR
# clf4 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf4.fit(X_train, y_train)
#
# # # print(X_train_male['sex'])
# # import matplotlib.pyplot as plt
# # y = np.arange(len(allBFdataset.columns)-1)
# # plt.barh(y,clf4.coef_[0])
# # plt.yticks(y,allBFdataset.columns)
# # plt.show()
#
# # print(clf4.coef_[0])

dataset_orig = resetDataset()
print(dataset_orig.shape)
print(dataset_orig[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week', 'Probability']].head(30))

for index, row in dataset_orig.iterrows():
    row = [row.values[0:len(row.values) - 1]]
    y_male_white = clf1.predict(row)
    y_male_black = clf2.predict(row)
    y_female_white = clf3.predict(row)
    y_female_black = clf4.predict(row)
    if not ((y_male_white[0] == y_male_black[0]) and (y_male_black[0] == y_female_white[0]) and (
            y_female_white[0] == y_female_black[0])):
        dataset_orig = dataset_orig.drop(index)
#     else:
#         print(y_male_white[0],y_male_black[0],y_female_white[0],y_female_black[0])

print(dataset_orig.shape)
print(dataset_orig[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week', 'Probability']].head(30))