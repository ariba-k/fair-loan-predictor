import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath('..'))
fileloc = str(sys.path[0]) + '\\' + 'HMDACT.csv'

dataset_orig = pd.read_csv(fileloc, dtype=object)
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
dataset_orig.loc[(dataset_orig.action_taken == '1'), 'action_taken'] = 1
dataset_orig.loc[(dataset_orig.action_taken == '2'), 'action_taken'] = 0
dataset_orig.loc[(dataset_orig.action_taken == '3'), 'action_taken'] = 0

##-------------------NA Removal------------------
def removeNA(array_columns):
    for startIndex in range(len(array_columns)):
        currentIndexName = dataset_orig[dataset_orig[array_columns[startIndex]] == "NA"].index
        dataset_orig.drop(currentIndexName, inplace=True)


def removeExempt(array_columns):
    for startIndex in range(len(array_columns)):
        currentIndexName = dataset_orig[dataset_orig[array_columns[startIndex]] == "Exempt"].index
        dataset_orig.drop(currentIndexName, inplace=True)


def removeNan(array_columns):
    for startIndex in range(len(array_columns)):
        currentIndexName = dataset_orig[dataset_orig[array_columns[startIndex]] == ""].index
        dataset_orig.drop(currentIndexName, inplace=True)


def removeBlank(array_columns):
    for startIndex in range(len(array_columns)):
        currentIndexName = dataset_orig[dataset_orig[array_columns[startIndex]] == ""].index
        dataset_orig.drop(currentIndexName, inplace=True)


def bucketingColumns(column, arrayOfUniqueVals, nicheVar):
    currentCol = column
    for firstIndex in range(len(arrayOfUniqueVals)):
        try:
            dataset_orig.loc[(nicheVar == arrayOfUniqueVals[firstIndex]), currentCol] = firstIndex
        except:
            print("This number didn't work:\n", firstIndex)


####---------------Reset Indexes----------------
dataset_orig.reset_index(drop=True, inplace=True)
# dataset_orig.loc[(dataset_orig.derived_race == 'Joint'),'derived_race']='Joint2'
# dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Joint'),'derived_ethnicity']='Joint1'
###----------------Begin Code------------------
# print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(70))
####################################################################################################################################
dataset_orig = dataset_orig.drop(
    ['census_tract', 'activity_year', 'lei', 'state_code', 'county_code', 'conforming_loan_limit',
     'derived_dwelling_category', 'loan_to_value_ratio', 'interest_rate',
     'rate_spread', 'total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points',
     'lender_credits', 'loan_term', 'prepayment_penalty_term', 'intro_rate_period', 'property_value',
     'multifamily_affordable_units', 'debt_to_income_ratio', 'applicant_ethnicity-2',
     'applicant_ethnicity-3', 'applicant_ethnicity-4', 'applicant_ethnicity-5', 'co-applicant_ethnicity-2',
     'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5', 'applicant_race-2',
     'applicant_race-3', 'applicant_race-4', 'applicant_race-5', 'co-applicant_race-2', 'co-applicant_race-3',
     'co-applicant_race-4', 'co-applicant_race-5', 'applicant_age_above_62', 'co-applicant_age_above_62', 'aus-2',
     'aus-3', 'aus-4', 'aus-5', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4', 'total_units',
     "applicant_age", "co-applicant_age"], axis=1)
# removed: rate of spread; income; county_code, state_code, activity_year, dervived_mba
removeNA(list(dataset_orig.columns))
removeExempt(list(dataset_orig.columns))
removeBlank(list(dataset_orig.columns))
dataset_orig.reset_index(drop=True, inplace=True)

# dataset_orig.loc[(dataset_orig.activity_year == '2020'), 'activity_year'] = 1
#
# dataset_orig.loc[(dataset_orig.state_code == state_string), 'state_code'] = 0

####---------------Bucketing-------------------
bucketingColumns('derived_loan_product_type',
                 ['Conventional:First Lien', 'FHA:First Lien', 'VA:First Lien', 'FSA/RHS:First Lien',
                  'Conventional:Subordinate Lien', 'FHA:Subordinate Lien', 'VA:Subordinate Lien',
                  'FSA/RHS:Subordinate Lien'], dataset_orig.derived_loan_product_type)

## Change symbolics to numerics
dataset_orig.loc[(dataset_orig.derived_sex == 'Female'), 'derived_sex'] = 0
dataset_orig.loc[(dataset_orig.derived_sex == 'Male'), 'derived_sex'] = 1
dataset_orig.loc[(dataset_orig.derived_sex == 'Joint'), 'derived_sex'] = 2
dataset_orig.loc[(dataset_orig.derived_race == 'Black or African American'), 'derived_race'] = 0
dataset_orig.loc[(dataset_orig.derived_race == 'White'), 'derived_race'] = 1
dataset_orig.loc[(dataset_orig.derived_race == 'Joint'), 'derived_race'] = 2
dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Hispanic or Latino'), 'derived_ethnicity'] = 0
dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Not Hispanic or Latino'), 'derived_ethnicity'] = 1
dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Joint'), 'derived_ethnicity'] = 2

dataset_orig = dataset_orig.apply(pd.to_numeric)
dataset_orig = dataset_orig.dropna()
removeNA(list(dataset_orig.columns))
float_col = dataset_orig.select_dtypes(include=['float64'])

for col in float_col.columns.values:
    dataset_orig[col] = dataset_orig[col].astype('int64')



dataset_orig.reset_index(drop=True, inplace=True)

# D is the dataset that we are using--in other words, HMDA_df

print("Before balancing this is the shape:", dataset_orig.shape)
dataset_orig.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\BeforeBalancingDataset.csv')

numCols = len(dataset_orig.columns) - 1
numDeleted = 0
threshold = 72058


while(numDeleted < threshold):
    numRows = len(dataset_orig) - 1
    numRandom = random.randint(0, numRows)
    # print(numRandom)
    randomRowActionTaken = dataset_orig.loc[numRandom].iat[numCols]


    if(randomRowActionTaken == 0):
        dataset_orig = dataset_orig.drop(numRandom)
        dataset_orig.reset_index(drop=True, inplace=True)
        numDeleted = numDeleted + 1
        print(numDeleted)

dataset_orig.reset_index(drop=True, inplace=True)

fileToSaveTo = str(sys.path[0]) + '\\' + 'ProcessedCTHMDA.csv'

dataset_orig.to_csv(fileToSaveTo)
# print("After balancing this is the shape:", dataset_orig.shape)