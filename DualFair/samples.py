# -------------------Imports---------------------------
import os
import sys
import numpy as np
import pandas as pd
import statistics

from statistics import median
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

from SMOTE import smote
from Generate_Samples import generate_samples
from Delete_Samples import delete_samples

sys.path.append(os.path.abspath('../..'))


# Custom Exceptions
class EmptyList(Exception):
    pass


# Dataset Used Needs to be Large Enough to Have Data for all 27 Subsets

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = base_path + '/Data/raw_state_CT.csv'
input_file_1 = base_path + '/Data/HMDA_2020_Data.csv'
input_file_2 = base_path + '/Data/HMDA_2019_Data.csv'
input_file_3 = base_path + '/Data/HMDA_2018_Data.csv'
# interm_file = base_path + '\\Data\\FirstBalancedCA.csv'
final_file = base_path + '/Data/new_addition_file.csv'
# process_scale_file = base_path + '\\Data\\processedscaledCANOW.csv'
process_scale_file = base_path + '/Data/processedscaledCANOW.csv'
# other_file = base_path + '\\Data\\newDatasetOrig.csv'
other_file = base_path + '/Data/newDatasetOrig.csv'
# output_file = base_path + '\\Data\\DoubleBalancedCA.csv'
output_file = base_path + '/Data/DoubleBalancedCA.csv'

print("I'M INPUT:", input_file_1)
dataset_orig = pd.read_csv(input_file_1, dtype=object).sample(n=10000000)
print('ROWS:', dataset_orig.shape[0])
# df_2020 = pd.read_csv(input_file_1, dtype=object).sample(n=5000000)
# df_2019 = pd.read_csv(input_file_2, dtype=object).sample(n=5000000)
# df_2018 = pd.read_csv(input_file_3, dtype=object).sample(n=5000000)
# dataset_orig = pd.concat([df_2020, df_2019, df_2018])
# dataset_orig = dataset_orig.sample(frac=1)
# dataset_orig.reset_index(drop=True, inplace=True)

print('Data', dataset_orig.shape)
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

# switch action_taken to last column
action_taken_col = dataset_orig.pop('action_taken')
dataset_orig.insert(len(dataset_orig.columns), 'action_taken', action_taken_col)


#####-----------------------Declaring Cutting Functions-----------------
def removeExempt(array_columns, df):
    for startIndex in range(len(array_columns)):
        currentIndexName = df[df[array_columns[startIndex]] == "Exempt"].index
        df.drop(currentIndexName, inplace=True)


def removeBlank(array_columns, df):
    for startIndex in range(len(array_columns)):
        currentIndexName = df[df[array_columns[startIndex]] == ""].index
        df.drop(currentIndexName, inplace=True)


def bucketingColumns(column, arrayOfUniqueVals, nicheVar):
    currentCol = column
    for firstIndex in range(len(arrayOfUniqueVals)):
        try:
            dataset_orig.loc[(nicheVar == arrayOfUniqueVals[firstIndex]), currentCol] = firstIndex
        except:
            print("This number didn't work:\n", firstIndex)


#####------------------Scaling------------------------------------
def scale_dataset(processed_df):
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)
    return scaled_df


###------------------Preprocessing Function (includes Scaling)------------------------
def preprocessing(dataset_orig):
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[
        ['derived_msa-md', 'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex',
         'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage',
         'interest_rate', 'loan_to_value_ratio',
         'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status',
         'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features',
         'construction_method',
         'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
         'applicant_credit_score_type',
         'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
         'applicant_ethnicity_observed',
         'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed',
         'co-applicant_race_observed',
         'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed',
         'submission_of_application',
         'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population',
         'tract_minority_population_percent',
         'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
         'tract_one_to_four_family_homes',
         'tract_median_age_of_housing_units', 'action_taken']]

    # Below we are taking out rows in the dataset with values we do not care for. This is from lines 23 - 99.
    ###--------------------Sex------------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_sex'] == 'Male') |
                                (dataset_orig['derived_sex'] == 'Female') |
                                (dataset_orig['derived_sex'] == 'Joint')]
    dataset_orig['derived_sex'] = dataset_orig['derived_sex'].replace(['Female', 'Male', 'Joint'],
                                                                      [0, 1, 2])
    print('sex: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ###-------------------Races-----------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_race'] == 'White') |
                                (dataset_orig['derived_race'] == 'Black or African American') |
                                (dataset_orig['derived_race'] == 'Joint')]
    dataset_orig['derived_race'] = dataset_orig['derived_race'].replace(['Black or African American', 'White', 'Joint'],
                                                                        [0, 1, 2])
    print('race: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ####----------------Ethnicity-------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_ethnicity'] == 'Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Not Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Joint')]
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(
        ['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
        [0, 1, 2])
    print('ethnicity: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    # ----------------Action_Taken-----------------
    dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
                                (dataset_orig['action_taken'] == '2') |
                                (dataset_orig['action_taken'] == '3')]

    dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
                                                                        [1, 0, 0])

    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ######----------------Loan Product-------------------
    # assigns each unique categorical value a unique integer id
    dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes

    print('loan product: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    # #######----------Remove Exempts and Blanks-------------
    # array_columns_to_remove = ["interest_rate", 'loan_to_value_ratio']
    # removeExempt(array_columns_to_remove, dataset_orig)
    # removeBlank(array_columns_to_remove, dataset_orig)

    ####---------------Scale Dataset---------------

    array_columns_to_remove = ['interest_rate', 'loan_to_value_ratio']

    removeExempt(array_columns_to_remove, dataset_orig)
    removeBlank(array_columns_to_remove, dataset_orig)
    print(dataset_orig[['interest_rate', 'loan_to_value_ratio']])
    dataset_orig = dataset_orig.apply(pd.to_numeric)
    dataset_orig = dataset_orig.dropna()
    dataset_orig = scale_dataset(dataset_orig)
    print(dataset_orig[['interest_rate', 'loan_to_value_ratio']])

    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


###---------Call Preprocessing to Create Processed_Scaled_Df---------------
processed_scaled_df = preprocessing(dataset_orig)
processed_scaled_shape = processed_scaled_df.shape
processed_scaled_df.to_csv(process_scale_file, index=False)

##------------------Check beginning Measures----------------------

processed_scaled_df["derived_sex"] = pd.to_numeric(processed_scaled_df.derived_sex, errors='coerce')
processed_scaled_df["derived_race"] = pd.to_numeric(processed_scaled_df.derived_race, errors='coerce')
processed_scaled_df["derived_ethnicity"] = pd.to_numeric(processed_scaled_df.derived_ethnicity, errors='coerce')
processed_scaled_df["action_taken"] = pd.to_numeric(processed_scaled_df.action_taken, errors='coerce')

joint_df = processed_scaled_df[(processed_scaled_df['derived_ethnicity'] == 1) & (processed_scaled_df['derived_race'] == 1) & (processed_scaled_df['derived_sex'] == 0.5) & (processed_scaled_df['action_taken'] == 0)]
# filterinfDataframeFemale = processed_scaled_df[(processed_scaled_df['derived_ethnicity'] == 0) & (processed_scaled_df['derived_race'] == 1) & (processed_scaled_df['derived_sex'] == 0) & (processed_scaled_df['action_taken'] == 0)]
print(joint_df[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head())
# print(filterinfDataframeFemale[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head())
# comb_df = pd.concat([filterinfDataframeMale, filterinfDataframeFemale])
joint_df.to_csv(final_file, index=False)