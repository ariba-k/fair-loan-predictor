import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


sys.path.append(os.path.abspath('..'))
fileloc = str(sys.path[0]) + '\\' + 'TheDebiasedDataset.csv'

dataset_orig = pd.read_csv(fileloc, dtype=object)
dataset_orig.reset_index(drop=True, inplace=True)

dataset_orig = dataset_orig.apply(pd.to_numeric)

print("Before balancing this is the shape:", dataset_orig.shape)

numCols = len(dataset_orig.columns) - 1
numDeleted = 0
threshold = 4164


while(numDeleted < threshold):
    x = 0
    numRows = len(dataset_orig) - 1
    numRandom = random.randint(0, numRows)
    # print(numRandom)
    randomRowActionTaken = dataset_orig.loc[numRandom].iat[numCols]
    print(randomRowActionTaken)
    if(randomRowActionTaken == 1):
        dataset_orig = dataset_orig.drop(numRandom)
        dataset_orig.reset_index(drop=True, inplace=True)
        numDeleted = numDeleted + 1
        print('NumDeleted Val:', numDeleted)

dataset_orig.reset_index(drop=True, inplace=True)

print("After balancing this is the shape:", dataset_orig.shape)
fileToSaveTo = str(sys.path[0]) + '\\' + 'BalancedTheDebiasedDataset.csv'

dataset_orig.to_csv(fileToSaveTo)
# print("After balancing this is the shape:", dataset_orig.shape)