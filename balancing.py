import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

dataset_orig = pd.read_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\MEHMDA.csv',
                           dtype=object)
print(dataset_orig.shape)
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
dataset_orig.loc[(dataset_orig.action_taken == '1'), 'action_taken'] = 0
dataset_orig.loc[(dataset_orig.action_taken == '2'), 'action_taken'] = 1
dataset_orig.loc[(dataset_orig.action_taken == '3'), 'action_taken'] = 1



dataset_orig.reset_index(drop=True, inplace=True)

# D is the dataset that we are using--in other words, HMDA_df

print("Before balancing this is the shape:", dataset_orig.shape)

numCols = len(dataset_orig.columns) - 1
numDeleted = 0
threshold = 24300

while(numDeleted < threshold):
    numRows = len(dataset_orig) - 1
    numRandom = random.randint(0, numRows)
    # print(numRandom)
    randomRowActionTaken = dataset_orig.loc[numRandom].iat[numCols]

    print("Action:", randomRowActionTaken)
    if(randomRowActionTaken == 0):
        dataset_orig = dataset_orig.drop(numRandom)
        dataset_orig.reset_index(drop=True, inplace=True)
        numDeleted = numDeleted + 1
        print(numDeleted)

dataset_orig.reset_index(drop=True, inplace=True)
dataset_orig.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\BalancedMEHMDA.csv')
print("After balancing this is the shape:", dataset_orig.shape)