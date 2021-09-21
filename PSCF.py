import numpy as np
import pandas as pd
import os
import sys

import numpy as np
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

sys.path.append(os.path.abspath('..'))
###======================Part 1: Importing Datasets Begins======================
base_path = str(sys.path[0])

input_file = base_path + '\\Data\\raw_state_CA.csv'
interm_file = base_path + '\\Data\\FirstBalancedCA.csv'
output_file = base_path + '\\Data\\DoubleBalancedCA.csv'


dataset_orig = pd.read_csv(input_file, dtype=object)
print('Data', dataset_orig.shape)
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])


nonbiased_features = ["activity_year", "tract_median_age_of_housing_units", "tract_one_to_four_family_homes", "tract_owner_occupied_units", "tract_population", "initially_payable_to_institution", "submission_of_application", "co-applicant_credit_score_type", "applicant_credit_score_type", "multifamily_affordable_units", "total_units", "other_nonamortizing_features", "balloon_payment", "interest_only_payment", "negative_amortization"]

###======================Part 2: Using PSCF Begins================================
#Make sure the list of nonbiased featueres also contains the label
def PSCF(list_of_nonbiased_features, dataset_orig=dataset_orig):
    dataset_orig_nonbiased_features = dataset_orig[[list_of_nonbiased_features]]
    dataset_orig_nonbiased_features_train, dataset_orig_nonbiased_features_test = train_test_split(dataset_orig_nonbiased_features, test_size=0.3,
                                                                         random_state=0, shuffle=True)

    X_train, y_train = dataset_orig_nonbiased_features_train.loc[:, dataset_orig_nonbiased_features_train.columns != 'action_taken'], \
                       dataset_orig_nonbiased_features_train['action_taken']
    X_test, y_test = dataset_orig_nonbiased_features_test.loc[:, dataset_orig_nonbiased_features_test.columns != 'action_taken'], \
                     dataset_orig_nonbiased_features_test['action_taken']

    # --- LSR
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


final_clf, X_test, y_test = PSCF(nonbiased_features)