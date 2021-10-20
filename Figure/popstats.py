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

input_file = r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\Data\raw_state_WY.csv'
interm_file = base_path + '\\Data\\FirstBalancedCA.csv'
output_file = base_path + '\\Data\\DoubleBalancedCA.csv'


dataset_orig = pd.read_csv(input_file, dtype=object)
action_taken_col = dataset_orig.pop('action_taken')
dataset_orig.insert(len(dataset_orig.columns), 'action_taken', action_taken_col)
print('Data', dataset_orig.shape)
print(dataset_orig[[ 'income', 'derived_sex', 'action_taken']])



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



def preprocessing(dataset_orig):
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[['derived_msa-md', 'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex',
     'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage', 'income', 'interest_rate',
     'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status',
     'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'construction_method',
     'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest','applicant_credit_score_type',
     'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_ethnicity_observed',
     'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed','co-applicant_race_observed',
     'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed','submission_of_application',
     'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population','tract_minority_population_percent',
     'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units','tract_one_to_four_family_homes',
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
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
                                                                                  [0, 1, 2])
    print('ethnicity: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    #----------------Action_Taken-----------------
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


    ##----------------Solve NAN problem-----------

    array_columns_to_remove = ["interest_rate"]
    print("THIS IS ME", dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'interest_rate', 'action_taken']].head(50))
    removeExempt(array_columns_to_remove, dataset_orig)
    removeBlank(array_columns_to_remove, dataset_orig)
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'interest_rate', 'action_taken']].head(50))
    dataset_orig = dataset_orig.apply(pd.to_numeric)
    dataset_orig = dataset_orig.dropna()
    float_col = dataset_orig.select_dtypes(include=['float64'])

    # for col in float_col.columns.values:
    #     dataset_orig[col] = dataset_orig[col].astype('int64')
    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig

#==============================splitting datasets for getting stats==========================


def get_unique_df(processed_scaled_df,ind_cols):
    uniques = [processed_scaled_df[i].unique().tolist() for i in ind_cols]
    unique_df = pd.DataFrame(product(*uniques), columns=ind_cols)
    return unique_df


def split_dataset(processed_scaled_df, ind_cols):
    unique_df = get_unique_df(processed_scaled_df, ind_cols)
    unique_df.drop(unique_df.index[(unique_df[ind_cols[0]] == 2)], axis=0, inplace=True)
    unique_df.reset_index(drop=True, inplace=True)
    print(unique_df)
    combination_df = [pd.merge(processed_scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner') for i in
                      range(unique_df.shape[0])]

    return combination_df, unique_df


processed_dataset = preprocessing(dataset_orig)
ind_cols = ['derived_sex']
dep_col = 'action_taken'
comb_gender_df, unique_gender_df = split_dataset(processed_dataset, ind_cols)
print('gender:', comb_gender_df[1][["derived_sex", "income"]])

ind_cols = ['derived_ethnicity']
dep_col = 'action_taken'
comb_ethnic_df, unique_ethnic_df = split_dataset(processed_dataset, ind_cols)
print('ethnic:', comb_ethnic_df[1][["derived_ethnicity", "income"]])

ind_cols = ['derived_race']
dep_col = 'action_taken'
comb_race_df, unique_race_df = split_dataset(processed_dataset, ind_cols)
print('race:', comb_race_df[1][["derived_race", "income"]])


array_col = ["income", "loan_amount", "interest_rate"] #HYPER

def getMeans(array_col_means, df):
    for col in range(len(array_col_means)):
        print("MEAN:", array_col_means[col], df[array_col_means[col]].mean())

def getMedians(array_col_medians, df):
    for col in range(len(array_col_medians)):
        print("MEDIAN", array_col_medians[col], df[array_col_medians[col]].median())

def evaluteMeansandMedians(comb_df):
    for df in comb_df:
        print(df.shape[0])
        getMeans(array_col, df)
        getMedians(array_col, df)


evaluteMeansandMedians(comb_gender_df)
print('///////////////////////////////////////////////////////////////////////')
evaluteMeansandMedians(comb_ethnic_df)
print('///////////////////////////////////////////////////////////////////////')
evaluteMeansandMedians(comb_race_df)