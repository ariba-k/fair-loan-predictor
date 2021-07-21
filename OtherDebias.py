## Here I am dividing the data first based onto protected attribute value and then train two separate models

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

## Load dataset
dataset_orig = pd.read_csv('../dataset/adult.data.csv')

## Drop NULL values
dataset_orig = dataset_orig.dropna()

## Drop categorical features
dataset_orig = dataset_orig.drop(['workclass','fnlwgt','education','marital-status','occupation','relationship','native-country'],axis=1)

## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)


## Discretize age
dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 60 ) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 50 ) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 40 ) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 30 ) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 20 ) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 10 ) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)



# divide the data based on sex
dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig.groupby(dataset_orig['sex'] == 0)]

print(dataset_orig_male.shape)
print(dataset_orig_female.shape)