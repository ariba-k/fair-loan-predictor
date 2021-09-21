import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

sys.path.append(os.path.abspath('..'))

'''This is the first script in the debiasing pipeline, and, as such, you should always run this first if 
you are trying to debias a random CSV from HMDA. This script will balance your said dataset intial and 
change some categories into numeric values. To do all these things, you want to change the CSV name
on line 15 to your said csv. Lastly, you can save the balanced dataset using line 173. The program
will automatically do these things. IT IS IMPORTANT TO NOTE, THOUGH, YOU MUST PERSONALLY CHANGE 
THE THRESHOLD NUMBER ON LINE 164 (this is your approved - rejected people in the CSV)'''

fileloc = str(sys.path[0]) + '\\Data\\' + 'raw_state_CA.csv'

dataset_orig = pd.read_csv(fileloc, dtype=object)
print(dataset_orig.shape)

dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
                            (dataset_orig['action_taken'] == '2') |
                            (dataset_orig['action_taken'] == '3')]

dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
                                                                    [1, 0, 0])



###----------------Balancing-----------------

def balance(dataset_orig):
    if dataset_orig.empty:
        return dataset_orig
    print('imbalanced data:\n', dataset_orig['action_taken'].value_counts())
    action_df = dataset_orig['action_taken'].value_counts()
    maj_label = action_df.index[0]
    min_label = action_df.index[-1]
    df_majority = dataset_orig[dataset_orig.action_taken == maj_label]
    df_minority = dataset_orig[dataset_orig.action_taken == min_label]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority.index),  # to match minority class
                                       random_state=123)
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    df_downsampled.reset_index(drop=True, inplace=True)

    print('balanced data:\n', df_downsampled['action_taken'].value_counts())

    print('processed data: ' + str(df_downsampled.shape))

    return df_downsampled


balanced_df = balance(dataset_orig)

fileToSaveTo = str(sys.path[0]) + '\\Data\\' + 'balanced_state_CA.csv'
balanced_df.to_csv(fileToSaveTo)
# print("After balancing this is the shape:", dataset_orig.shape)