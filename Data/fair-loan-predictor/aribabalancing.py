import os
import sys
import pandas as pd
from sklearn.utils import resample

sys.path.append(os.path.abspath('..'))

'''This is the first script in the debiasing pipeline, and, as such, you should always run this first if 
you are trying to debias a random CSV from HMDA. This script will balance your said dataset initial and 
change some categories into numeric values. To do all these things, you want to change the CSV name
on line 15 to your said csv. Lastly, you can save the balanced dataset using line 173. The program
will automatically do these things. '''

# str(sys.path[0]) + '\\Data\\' + 'HMDACT.csv'
input_file = str(sys.path[0]) + '\\Data\\WYHMDA.csv'
output_file = str(sys.path[0]) + '\\Data\\processed_state_WY.csv'

# str(sys.path[0]) + '\\Data\\' + 'ProcessedCTHMDA.csv'

dataset_orig = pd.read_csv(input_file, dtype=object)


def preprocessing(dataset_orig):
    print('original data: ' + str(dataset_orig.shape))
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']]
    print('column selection: ' + str(dataset_orig.shape))

    # Below we are taking out rows in the dataset with values we do not care for. This is from lines 23 - 99.
    ###--------------------Sex------------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_sex'] == 'Male') |
                                (dataset_orig['derived_sex'] == 'Female') |
                                (dataset_orig['derived_sex'] == 'Joint')]
    dataset_orig['derived_sex'] = dataset_orig['derived_sex'].replace(['Female', 'Male', 'Joint'],
                                                                      [0, 1, 2])
    print('sex: ' + str(dataset_orig.shape))
    ###-------------------Races-----------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_race'] == 'White') |
                                (dataset_orig['derived_race'] == 'Black or African American') |
                                (dataset_orig['derived_race'] == 'Joint')]
    dataset_orig['derived_race'] = dataset_orig['derived_race'].replace(['Black or African American', 'White', 'Joint'],
                                                                        [0, 1, 2])
    print('race: ' + str(dataset_orig.shape))
    ####----------------Ethnicity-------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_ethnicity'] == 'Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Not Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Joint')]
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(['Hispanic or Latino',
                                                                                   'Not Hispanic or Latino', 'Joint'],
                                                                                  [0, 1, 2])
    print('ethnicity: ' + str(dataset_orig.shape))
    ###----------------Action_Taken-----------------
    dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
                                (dataset_orig['action_taken'] == '2') |
                                (dataset_orig['action_taken'] == '3')]

    dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
                                                                        [1, 0, 0])
    print('action taken: ' + str(dataset_orig.shape))

    # ######----------------Loan Product-------------------
    # # assigns each unique categorical value a unique integer id
    # dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes
    #
    # print('loan product: ' + str(dataset_orig.shape))

    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


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


processed_df = preprocessing(dataset_orig)
# df_downsampled = balance(dataset_orig)
#
processed_df.to_csv(output_file, index=False)
