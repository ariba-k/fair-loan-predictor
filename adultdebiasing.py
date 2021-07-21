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
    dataset_orig = pd.read_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\adult.csv')
    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()
    ## Drop categorical features
    dataset_orig = dataset_orig.drop(
        ['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'], axis=1)
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
print('Printing all female \n', allFemaleDataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

allMaleDataset = splittingDataset('sex', [0])
print('Printing all male \n', allMaleDataset[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(50))

allBFdataset = splittingDatasetSecondLayer('')
# divide the data based on sex

dataset_orig = resetDataset()

dataset_orig_male, dataset_orig_female = [x for _, x in dataset_orig.groupby(dataset_orig['sex'] == 0)]
print(dataset_orig_male[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(20))
print(dataset_orig_female[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(20))



# # divide the data based on race
# dataset_orig_male_white, dataset_orig_male_black = [x for _, x in dataset_orig_male.groupby(dataset_orig['race'] == 0)]
# dataset_orig_female_white, dataset_orig_female_black = [x for _, x in dataset_orig_female.groupby(dataset_orig['race'] == 0)]
# dataset_orig_male['sex'] = 0
# X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
# # --- LSR
# clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf_male.fit(X_train_male, y_train_male)
# # print(X_train_male['sex'])
# import matplotlib.pyplot as plt
# y = np.arange(len(dataset_orig_male.columns)-1)
# plt.barh(y,clf_male.coef_[0])
# X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
# # --- LSR
# clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf_female.fit(X_train_female, y_train_female)
# y = np.arange(len(dataset_orig_female.columns)-1)
# plt.barh(y,clf_female.coef_[0])
# # print(dataset_orig.shape)
# df_removed = pd.DataFrame(columns=dataset_orig.columns)
# for index, row in dataset_orig.iterrows():
#     row_ = [row.values[0:len(row.values) - 1]]
#     y_male = clf_male.predict(row_)
#     y_female = clf_female.predict(row_)
#     if y_male[0] != y_female[0]:
#         df_removed = df_removed.append(row, ignore_index=True)
#         dataset_orig = dataset_orig.drop(index)
#
# print(dataset_orig[['age', 'education-num', 'race', 'sex', 'capital-loss', 'hours-per-week']].head(20))
# print(df_removed.shape)