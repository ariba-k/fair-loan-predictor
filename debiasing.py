## Here I am dividing the data first based onto protected attribute value and then train two separate models
import os
import sys
from matplotlib.pyplot import yticks, show, barh
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath('..'))
# from Measure import measure_final_score
sys.path.append(os.path.abspath('..'))


##----KEY FUNCTIONS----##
# ==========================ABOVE IMPORTS========================================
#:Training dataset D, Sensitive attribute S, Binary
# classification model M trained on D, Input space
# similarity threshold delta
def resetDataset():
    state_string = 'CA'
    dataset_orig = pd.read_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\WYHMDA.csv',
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



    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)
    # dataset_orig.loc[(dataset_orig.derived_race == 'Joint'),'derived_race']='Joint2'
    # dataset_orig.loc[(dataset_orig.derived_ethnicity == 'Joint'),'derived_ethnicity']='Joint1'
    ###----------------Begin Code------------------
    # print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(70))
    ####################################################################################################################################
    dataset_orig = dataset_orig.drop(['census_tract', 'activity_year', 'lei', 'derived_msa-md', 'state_code', 'county_code', 'conforming_loan_limit',
    'derived_loan_product_type', 'derived_dwelling_category', 'loan_to_value_ratio', 'interest_rate',
    'rate_spread', 'total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points',
    'lender_credits', 'loan_term', 'prepayment_penalty_term', 'intro_rate_period', 'property_value',
    'multifamily_affordable_units', 'debt_to_income_ratio', 'applicant_ethnicity-2',
    'applicant_ethnicity-3', 'applicant_ethnicity-4', 'applicant_ethnicity-5', 'co-applicant_ethnicity-2',
    'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5', 'applicant_race-2',
    'applicant_race-3', 'applicant_race-4', 'applicant_race-5', 'co-applicant_race-2', 'co-applicant_race-3',
    'co-applicant_race-4', 'co-applicant_race-5', 'applicant_age_above_62', 'co-applicant_age_above_62', 'aus-2',
    'aus-3', 'aus-4', 'aus-5', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4', 'total_units',
    "applicant_age", "co-applicant_age"], axis=1)
    #removed: rate of spread; income; county_code, state_code, activity_year, dervived_mba
    removeNA(list(dataset_orig.columns))
    removeExempt(list(dataset_orig.columns))
    removeBlank(list(dataset_orig.columns))
    dataset_orig.reset_index(drop=True, inplace=True)

    # dataset_orig.loc[(dataset_orig.activity_year == '2019'), 'activity_year'] = 0
    # dataset_orig.loc[(dataset_orig.activity_year == '2020'), 'activity_year'] = 1
    #
    # dataset_orig.loc[(dataset_orig.state_code == state_string), 'state_code'] = 0


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



    # scaler = MinMaxScaler()
    # dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    # print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
    # divide the data based on sex
    # dataset_new = dataset_orig.groupby(dataset_orig['derived_sex'] == 0)
    # print(dataset_new[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
    return dataset_orig


# dataset_orig = resetDataset()
# print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
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




# =============================FIRST LAYER DIVIDE=====================================
allFemaleDataset = splittingDataset('derived_sex', [.5, 1])
# print('Printing all female \n',
#       allFemaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allFemaleReset():
    allFemaleDataset = splittingDataset('derived_sex', [.5, 1])
    return allFemaleDataset


allMaleDataset = splittingDataset('derived_sex', [0, 1])
# print('Printing all male \n',
#       allMaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allMaleReset():
    allMaleDataset = splittingDataset('derived_sex', [0, 1])
    return allMaleDataset


allJointSexDataset = splittingDataset('derived_sex', [0, 0.5])
# print('Printing all Joint \n',
#       allJointSexDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allJointSexReset():
    allJointSexDataset = splittingDataset('derived_sex', [0, 0.5])
    return allJointSexDataset


# ===============================Second Layer Divide====================================
# First White and Black Females
allBFdataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allFemaleDataset)
# print('Printing all BFs \n',
#       allBFdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allBFReset():
    allFemaleDataset = allFemaleReset()
    allBFdataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allFemaleDataset)
    return allBFdataset


allFemaleDataset = allFemaleReset()
'''OKAY, ARASH ABOVE IS THE MAIN CHANGE; ESSENTIALLY, YOU HAVE TO RESET THE DATASET, WHICHEVER ONE YOU ARE USING,
BEFORE YOU USE IT FOR ANOTHER SPLIT; THIS IS WHY I DO THE BF SPLIT AND THEN RESET FOR THE WF SPLIT. I DO THIS BY
MAKING A FUNCTION CALLED allFemaleReset(). FYI, we are going to have to make a lot of functions, but I found
this is the most efficent method to use'''
allWFdataset = splittingDatasetSecondLayer('derived_race', [0, 1], allFemaleDataset)
# print('Printing all WFs \n',
#       allWFdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allWFReset():
    allFemaleDataset = allFemaleReset()
    allWFdataset = splittingDatasetSecondLayer('derived_race', [0, 1], allFemaleDataset)
    return allWFdataset


allFemaleDataset = allFemaleReset()
allJointRaceFemaleDataset = splittingDatasetSecondLayer('derived_race', [0, .5], allFemaleDataset)
# print('Printing all JointRaceFemales \n',
#       allJointRaceFemaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allJointRaceFemaleReset():
    allFemaleDataset = allFemaleReset()
    allJointRaceFemaleDataset = splittingDatasetSecondLayer('derived_race', [0, .5], allFemaleDataset)
    return allJointRaceFemaleDataset


# ---------------------------second white, black males--------------------------------------------------------------------------------------------
allBMdataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allMaleDataset)
# print('Printing all BMs \n',
#       allBMdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allBMReset():
    allMaleDataset = allMaleReset()
    allBMdataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allMaleDataset)
    return allBMdataset


allMaleDataset = allMaleReset()
allWMdataset = splittingDatasetSecondLayer('derived_race', [0, 1], allMaleDataset)
# print('Printing all WMs \n',
#       allWMdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allWMReset():
    allMaleDataset = allMaleReset()
    allWMdataset = splittingDatasetSecondLayer('derived_race', [0, 1], allMaleDataset)
    return allWMdataset


allMaleDataset = allMaleReset()
allJointRaceMaleDataset = splittingDatasetSecondLayer('derived_race', [0, .5], allMaleDataset)
# print('Printing all JointRaceMales \n',
#       allJointRaceMaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allJointRaceMaleReset():
    allMaleDataset = allMaleReset()
    allJointRaceMaleDataset = splittingDatasetSecondLayer('derived_race', [0, .5], allMaleDataset)
    return allJointRaceMaleDataset


# -------------------Third Joint Sex Races -----------------
allJointSexBlacksDataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allJointSexDataset)
# print('Printing all JointSexBlacks \n',
#       allJointSexBlacksDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allJointSexBlacksReset():
    allJointSexDataset = allJointSexReset()
    allJointSexBlacksDataset = splittingDatasetSecondLayer('derived_race', [.5, 1], allJointSexDataset)
    return allJointSexBlacksDataset


allJointSexDataset = allJointSexReset()
allJointSexWhitesDataset = splittingDatasetSecondLayer('derived_race', [0, 1], allJointSexDataset)
# print('Printing all JointSexWhites \n',
#       allJointSexWhitesDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allJointSexWhitesReset():
    allJointSexDataset = allJointSexReset()
    allJointSexWhitesDataset = splittingDatasetSecondLayer('derived_race', [0, 1], allJointSexDataset)
    return allJointSexWhitesDataset


allJointSexDataset = allJointSexReset()
allJointSexJointRaceDataset = splittingDatasetSecondLayer('derived_race', [0, .5], allJointSexDataset)
# print('Printing all JointSex and JointRaces \n',
#       allJointSexJointRaceDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allJointSexJointRaceReset():
    allJointSexDataset = allJointSexReset()
    allJointSexJointRaceDataset = splittingDatasetSecondLayer('derived_race', [0, .5], allJointSexDataset)
    return allJointSexJointRaceDataset


# ===============================Third Layer Divide====================================
# ---------------1. BF Ethnic Split --------------
allBFHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allBFdataset)
# print('Printing all BFHOL \n',
#       allBFHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBFdataset = allBFReset()
allBFNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allBFdataset)
# print('Printing all BFNHOLs \n',
#       allBFNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBFdataset = allBFReset()
allBFJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5], allBFdataset)
# print('Printing all BFJointEthnicitys \n',
#       allBFJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------2. WF Ethnic Split --------------
allWFHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allWFdataset)
# print('Printing all WFHOL \n',
#       allWFHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWFdataset = allWFReset()
allWFNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allWFdataset)
# print('Printing all WFNHOLs \n',
#       allWFNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWFdataset = allWFReset()
allWFJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5], allWFdataset)
# print('Printing all WFJointEthnicitys \n',
#       allWFJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------3. JointRaceFemale Ethnic Split --------------
allJointRaceFemaleHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allJointRaceFemaleDataset)
# print('Printing all JointRaceFemaleNHOL \n',
#       allJointRaceFemaleHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

allJointRaceFemaleDataset = allJointRaceFemaleReset()
allJointRaceFemaleNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allJointRaceFemaleDataset)
# print('Printing all JointRaceFemaleHOL \n',
#       allJointRaceFemaleNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointRaceFemaleDataset = allJointRaceFemaleReset()
allJointRaceFemaleJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5],
                                                                      allJointRaceFemaleDataset)
# print('Printing all JointRaceFemaleJointEthnicity \n', allJointRaceFemaleJointEthnicitydataset[
#     ['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------4. BM Ethnic Split --------------
allBMHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allBMdataset)
# print('Printing all BMHOL \n',
#       allBMHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBMdataset = allBMReset()
allBMNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allBMdataset)
# print('Printing all BMNHOLs \n',
#       allBMNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBMdataset = allBMReset()
allBMJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5], allBMdataset)
# print('Printing all BMJointEthnicitys \n',
#       allBMJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------5. WM Ethnic Split --------------
allWMHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allWMdataset)
# print('Printing all WMHOL \n',
#       allWMHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWMdataset = allWMReset()
allWMNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allWMdataset)
# print('Printing all WMNHOL \n',
#       allWMNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWMdataset = allWMReset()
allWMJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5], allWMdataset)
# print('Printing all WMJointEthnicity \n',
#       allWMJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------6. JointRaceMale Ethnic Split --------------
allJointRaceMaleHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allJointRaceMaleDataset)
# print('Printing all JointRaceMaleHOL \n',
#       allJointRaceMaleHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointRaceMaleDataset = allJointRaceMaleReset()
allJointRaceMaleNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allJointRaceMaleDataset)
# print('Printing all JointRaceMaleNHOL \n',
#       allJointRaceMaleNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointRaceMaleDataset = allJointRaceMaleReset()
allJointRaceMaleJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5],
                                                                    allJointRaceMaleDataset)
# print('Printing all JointRaceMaleJointEthnicity \n',
#       allJointRaceMaleJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(
#           50))
# ---------------7. JointSexBlacks Ethnic Split------------------
allJointSexBlacksHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allJointSexBlacksDataset)
# print('Printing all JointSexBlacksHOL \n',
#       allJointSexBlacksHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexBlacksDataset = allJointSexBlacksReset()
allJointSexBlacksNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allJointSexBlacksDataset)
# print('Printing all JointSexBlacksNHOL \n',
#       allJointSexBlacksNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexBlacksDataset = allJointSexBlacksReset()
allJointSexBlacksJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5],
                                                                     allJointSexBlacksDataset)
# print('Printing all JointSexBlacksJointEthnicity \n',
#       allJointSexBlacksJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(
#           50))
# ---------------8. JointSexWhites Ethnic Split------------------
allJointSexWhitesHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allJointSexWhitesDataset)
# print('Printing all JointSexWhitesHOL \n',
#       allJointSexWhitesHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexWhitesDataset = allJointSexWhitesReset()
allJointSexWhitesNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allJointSexWhitesDataset)
# print('Printing all JointSexWhitesNHOL \n',
#       allJointSexWhitesNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexWhitesDataset = allJointSexWhitesReset()
allJointSexWhitesJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5],
                                                                     allJointSexWhitesDataset)
# print('Printing all JointSexWhitesJointEthnicity \n',
#       allJointSexWhitesJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(
#           50))
# ---------------9. JointSexJointRace Ethnic Split------------------
allJointSexJointRaceHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [.5, 1], allJointSexJointRaceDataset)
# print('Printing all JointSexJointRaceHOL \n',
#       allJointSexJointRaceHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexJointRaceDataset = allJointSexJointRaceReset()
allJointSexJointRaceNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity', [0, 1], allJointSexJointRaceDataset)
# print('Printing all JointSexJointRaceNHOL \n',
#       allJointSexJointRaceNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexJointRaceDataset = allJointSexJointRaceReset()
allJointSexJointRaceJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [0, .5],
                                                                        allJointSexJointRaceDataset)
# print('Printing all JointSexJointRaceJointEthnicity \n', allJointSexJointRaceJointEthnicitydataset[
#     ['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


##################################
# Classifier Function (helps to build classifiers):

def createClassifier(D):
    D.reset_index(drop=True, inplace=True)
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

    D['derived_ethnicity'] = 0
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
    D['derived_race'] = 0
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
    D['derived_sex'] = 0
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
    numRows = len(D)
    numCols = len(D.columns) - 1
    hasZero = False
    hasOne = False
    for x in range(numRows):
      action_element = D.loc[x].iat[numCols]
      if(action_element == 0):
          hasZero = True
      if(action_element == 1):
          hasOne = True
    X_train, y_train = D.loc[:, D.columns != 'action_taken'], D['action_taken']
    # --- LSR
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=200)

    if numRows >= 2 and hasZero and hasOne:
        return clf.fit(X_train, y_train)
    else:
        return None



##################################
# --------------------Classifiers ---------------------
#THIS IS OUT OF ORDER -- PROBLEM?????
clf1 = createClassifier(allBFNHOLdataset)
clf2 = createClassifier(allBFHOLdataset)
clf3 = createClassifier(allBFJointEthnicitydataset)
clf4 = createClassifier(allWFNHOLdataset)
clf5 = createClassifier(allWFHOLdataset)
clf6 = createClassifier(allWFJointEthnicitydataset)
clf7 = createClassifier(allJointRaceFemaleNHOLdataset)
clf8 = createClassifier(allJointRaceFemaleHOLdataset)
clf9 = createClassifier(allJointRaceFemaleJointEthnicitydataset)
clf10 = createClassifier(allBMNHOLdataset)
clf11 = createClassifier(allBMHOLdataset)
clf12 = createClassifier(allBMJointEthnicitydataset)
clf13 = createClassifier(allWMNHOLdataset)
clf14 = createClassifier(allWMHOLdataset)
clf15 = createClassifier(allWMJointEthnicitydataset)
clf16 = createClassifier(allJointRaceMaleNHOLdataset)
clf17 = createClassifier(allJointRaceMaleHOLdataset)
clf18 = createClassifier(allJointRaceMaleJointEthnicitydataset)
clf19 = createClassifier(allJointSexBlacksNHOLdataset)
clf20 = createClassifier(allJointSexBlacksHOLdataset)
clf21 = createClassifier(allJointSexBlacksJointEthnicitydataset)
clf22 = createClassifier(allJointSexWhitesNHOLdataset)
clf23 = createClassifier(allJointSexWhitesHOLdataset)
clf24 = createClassifier(allJointSexWhitesJointEthnicitydataset)
clf25 = createClassifier(allJointSexJointRaceNHOLdataset)
clf26 = createClassifier(allJointSexJointRaceHOLdataset)
clf27 = createClassifier(allJointSexJointRaceJointEthnicitydataset)

# -----------Make Debiased Dataset (Remove Biased Points)------------
dataset_orig = resetDataset()
print(dataset_orig.shape)

for index, row in dataset_orig.iterrows():
    row = [row.values[0:len(row.values) - 1]]
    try:
        allBFNHOL_y = clf1.predict(row)
    except:
        allBFNHOL_y = None
    try:
        allBFHOL_y = clf2.predict(row)
    except:
        allBFHOL_y = None
    try:
        allBFJointEthnicity_y = clf3.predict(row)
    except:
        allBFJointEthnicity_y = None
    try:
        allWFNHOL_y = clf4.predict(row)
    except:
        allWFNHOL_y = None
    try:
        allWFHOL_y = clf5.predict(row)
    except:
        allWFHOL_y = None
    try:
        allWFJointEthnicity_y = clf6.predict(row)
    except:
        allWFJointEthnicity_y = None
    try:
        allJointRaceFemaleNHOL_y = clf7.predict(row)
    except:
        allJointRaceFemaleNHOL_y = None
    try:
        allJointRaceFemaleHOL_y = clf8.predict(row)
    except:
        allJointRaceFemaleHOL_y = None
    try:
        allJointRaceFemaleJointEthnicity_y = clf9.predict(row)
    except:
        allJointRaceFemaleJointEthnicity_y = None
    try:
        allBMNHOL_y = clf10.predict(row)
    except:
        allBMNHOL_y = None
    try:
        allBMHOL_y = clf11.predict(row)
    except:
        allBMHOL_y = None
    try:
        allBMJointEthnicity_y = clf12.predict(row)
    except:
        allBMJointEthnicity_y = None
    try:
        allWMNHOL_y = clf13.predict(row)
    except:
        allWMNHOL_y = None
    try:
        allWMHOL_y = clf14.predict(row)
    except:
        allWMHOL_y = None
    try:
        allWMJointEthnicity_y = clf15.predict(row)
    except:
        allWMJointEthnicity_y = None
    try:
        allJointRaceMaleNHOL_y = clf16.predict(row)
    except:
        allJointRaceMaleNHOL_y = None
    try:
        allJointRaceMaleHOL_y = clf17.predict(row)
    except:
        allJointRaceMaleHOL_y = None
    try:
        allJointRaceMaleJointEthnicity_y = clf18.predict(row)
    except:
        allJointRaceMaleJointEthnicity_y = None
    try:
        allJointSexBlacksNHOL_y = clf19.predict(row)
    except:
        allJointSexBlacksNHOL_y = None
    try:
        allJointSexBlacksHOL_y = clf20.predict(row)
    except:
        allJointSexBlacksHOL_y = None
    try:
        allJointSexBlacksJointEthnicity_y = clf21.predict(row)
    except:
        allJointSexBlacksJointEthnicity_y = None
    try:
        allJointSexWhitesNHOL_y = clf22.predict(row)
    except:
        allJointSexWhitesNHOL_y = None
    try:
        allJointSexWhitesHOL_y = clf23.predict(row)
    except:
        allJointSexWhitesHOL_y = None
    try:
        allJointSexWhitesJointEthnicity_y = clf24.predict(row)
    except:
        allJointSexWhitesJointEthnicity_y = None
    try:
        allJointSexJointRaceNHOL_y = clf25.predict(row)
    except:
        allJointSexJointRaceNHOL_y = None
    try:
        allJointSexJointRaceHOL_y = clf26.predict(row)
    except:
        allJointSexJointRaceHOL_y = None
    try:
        allJointSexJointRaceJointEthnicity_y = clf27.predict(row)
    except:
        allJointSexJointRaceJointEthnicity_y = None

    # allBFNHOL_y = clf1.predict(row)
    # allBFHOL_y = clf2.predict(row)
    # allBFJointEthnicity_y = clf3.predict(row)
    # allWFNHOL_y = clf4.predict(row)
    # allWFHOL_y = clf5.predict(row)
    # allWFJointEthnicity_y = clf6.predict(row)
    # allJointRaceFemaleNHOL_y = clf7.predict(row)
    # allJointRaceFemaleHOL_y = clf8.predict(row)
    # allJointRaceFemaleJointEthnicity_y = clf9.predict(row)
    # allBMNHOL_y = clf10.predict(row)
    # allBMHOL_y = clf11.predict(row)
    # allBMJointEthnicity_y = clf12.predict(row)
    # allWMNHOL_y = clf13.predict(row)
    # allWMHOL_y = clf14.predict(row)
    # allWMJointEthnicity_y = clf15.predict(row)
    # allJointRaceMaleNHOL_y = clf16.predict(row)
    # allJointRaceMaleHOL_y = clf17.predict(row)
    # allJointRaceMaleJointEthnicity_y = clf18.predict(row)
    # allJointSexBlacksNHOL_y = clf19.predict(row)
    # allJointSexBlacksHOL_y = clf20.predict(row)
    # allJointSexBlacksJointEthnicity_y = clf21.predict(row)
    # allJointSexWhitesNHOL_y = clf22.predict(row)
    # allJointSexWhitesHOL_y = clf23.predict(row)
    # allJointSexWhitesJointEthnicity_y = clf24.predict(row)
    # allJointSexJointRaceNHOL_y = clf25.predict(row)
    # allJointSexJointRaceHOL_y = clf26.predict(row)
    # allJointSexJointRaceJointEthnicity_y = clf27.predict(row)


        # try:
        #     allBFNHOL_y = clf1.predict(row)
        # except:
        #     allBFNHOL_y = None
        # try:
        #     allBFHOL_y = clf2.predict(row)
        # except:
        #     allBFHOL_y = None
        # try:
        #     allBFJointEthnicity_y = clf3.predict(row)
        # except:
        #     allBFJointEthnicity_y = None
        # try:
        #     allWFNHOL_y = clf4.predict(row)
        # except:
        #     allWFNHOL_y = None
        # try:
        #     allWFHOL_y = clf5.predict(row)
        # except:
        #     allWFHOL_y = None
        # try:
        #     allWFJointEthnicity_y = clf6.predict(row)
        # except:
        #     allWFJointEthnicity_y = None
        # try:
        #     allJointRaceFemaleNHOL_y = clf7.predict(row)
        # except:
        #     allJointRaceFemaleNHOL_y = None
        # try:
        #     allJointRaceFemaleHOL_y = clf8.predict(row)
        # except:
        #     allJointRaceFemaleHOL_y = None
        # try:
        #     allJointRaceFemaleJointEthnicity_y = clf9.predict(row)
        # except:
        #     allJointRaceFemaleJointEthnicity_y = None
        # try:
        #     allBMNHOL_y = clf10.predict(row)
        # except:
        #     allBMNHOL_y = None
        # try:
        #     allBMHOL_y = clf11.predict(row)
        # except:
        #     allBMHOL_y = None
        # try:
        #     allBMJointEthnicity_y = clf12.predict(row)
        # except:
        #     allBMJointEthnicity_y = None
        # try:
        #     allWMNHOL_y = clf13.predict(row)
        # except:
        #     allWMNHOL_y = None
        # try:
        #     allWMHOL_y = clf14.predict(row)
        # except:
        #     allWMHOL_y = None
        # try:
        #     allWMJointEthnicity_y = clf15.predict(row)
        # except:
        #     allWMJointEthnicity_y = None
        # try:
        #     allJointRaceMaleNHOL_y = clf16.predict(row)
        # except:
        #     allJointRaceMaleNHOL_y = None
        # try:
        #     allJointRaceMaleHOL_y = clf17.predict(row)
        # except:
        #     allJointRaceMaleHOL_y = None
        # try:
        #     allJointRaceMaleJointEthnicity_y = clf18.predict(row)
        # except:
        #     allJointRaceMaleJointEthnicity_y = None
        # try:
        #     allJointSexBlacksNHOL_y = clf19.predict(row)
        # except:
        #     allJointSexBlacksNHOL_y = None
        # try:
        #     allJointSexBlacksHOL_y = clf20.predict(row)
        # except:
        #     allJointSexBlacksHOL_y = None
        # try:
        #     allJointSexBlacksJointEthnicity_y = clf21.predict(row)
        # except:
        #     allJointSexBlacksJointEthnicity_y = None
        # try:
        #     allJointSexWhitesNHOL_y = clf22.predict(row)
        # except:
        #     allJointSexWhitesNHOL_y = None
        # try:
        #     allJointSexWhitesHOL_y = clf23.predict(row)
        # except:
        #     allJointSexWhitesHOL_y = None
        # try:
        #     allJointSexWhitesJointEthnicity_y = clf24.predict(row)
        # except:
        #     allJointSexWhitesJointEthnicity_y = None
        # try:
        #     allJointSexJointRaceNHOL_y = clf25.predict(row)
        # except:
        #     allJointSexJointRaceNHOL_y = None
        # try:
        #     allJointSexJointRaceHOL_y = clf26.predict(row)
        # except:
        #     allJointSexJointRaceHOL_y = None
        # try:
        #     allJointSexJointRaceJointEthnicity_y = clf27.predict(row)
        # except:
        #     allJointSexJointRaceJointEthnicity_y = None

    # print("Printing BFNHOL prediction:", allBFNHOL_y)
    # print("Printing allBFHOL_y prediction:",allBFHOL_y)
    # print("Printing allBFJointEthnicity_y prediction:", allBFJointEthnicity_y)
    # print("Printing allWFNHOL_y prediction:", allWFNHOL_y)
    # # print("Printing allWFHOL_y prediction:", allWFHOL_y)
    # print("Printing allWFJointEthnicity_y prediction:", allWFJointEthnicity_y)
    # print("Printing allJointRaceFemaleNHOL_y prediction:", allJointRaceFemaleNHOL_y)

    arrayClassifiers = [allBFNHOL_y, allBFHOL_y,
                        allBFJointEthnicity_y,
                        allWFNHOL_y,
                        allWFHOL_y,
                        allWFJointEthnicity_y,
                        allJointRaceFemaleNHOL_y,
                        allJointRaceFemaleHOL_y,
                        allJointRaceFemaleJointEthnicity_y,
                        allBMNHOL_y,
                        allBMHOL_y,
                        allBMJointEthnicity_y,
                        allWMNHOL_y,
                        allWMHOL_y,
                        allWMJointEthnicity_y,
                        allJointRaceMaleNHOL_y,
                        allJointRaceMaleHOL_y,
                        allJointRaceMaleJointEthnicity_y,
                        allJointSexBlacksNHOL_y,
                        allJointSexBlacksHOL_y,
                        allJointSexBlacksJointEthnicity_y,
                        allJointSexWhitesNHOL_y,
                        allJointSexWhitesHOL_y,
                        allJointSexWhitesJointEthnicity_y,
                        allJointSexJointRaceNHOL_y,
                        allJointSexJointRaceHOL_y,
                        allJointSexJointRaceJointEthnicity_y]
    for i in range((len(arrayClassifiers) - 1)):
        arrayConditional = []
        if arrayClassifiers[i] != None and arrayClassifiers[i + 1] != None:
            if not(arrayClassifiers[i][0] == arrayClassifiers[i + 1][0]):
                try:
                 dataset_orig = dataset_orig.drop(index)
                except:
                 jashan = "yes"
                 #print('Yeah, already deleted')
            # for i in range((len(arrayClassifiers) - 1)):
            #     arrayConditional = []
            #     if arrayClassifiers[i] != None and arrayClassifiers[i + 1] != None:
            #         arrayConditional.append(arrayClassifiers[i][0] == arrayClassifiers[i + 1][0])
            # for m in range((len(arrayConditional))):
            #     if (arrayConditional[m] == False):
            #         dataset_orig.drop(index, inplace=True)

print(dataset_orig.shape)
print(list(dataset_orig.columns))
print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex','action_taken']].head(50))
dataset_orig.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\DebiasedDataset.csv')

'''Scaling it is possibly causing the problem; 
    Is the loop running right?
    Why does my dataset size change when I include more variables (columns) 
    
    I think one of these will lead us to our answer: 
                 quadratic? CURRENTLY THIS IS THE ONE I AM WORKING ON; I don't think this it 
                 unbalancedness? 
                 Why are more rows deleted? (Promising)   ANSWER: NA and spaces are in the variables I include 
                 Why is it that even though I have more than enough data in each classifer they are giving me null values? (Promising)
                 
     Solving this error should give a solution:             
     Traceback (most recent call last):
       File "C:\\jasha\Documents\GitHub\fair-loan-predictor\debiasing.py", line 503, in <module>
       allBFNHOL_y = clf1.predict(row)
       AttributeError: 'NoneType' object has no attribute 'predict'
     
     Okay, no, here is the real fix and problem: 
     
     problem: the dataset being unbalanced (a 2-3-7-8 label is like 4,050/29,000); now combining this when you add more 
     varaibles the NA, spaces, and exempt rows are cut out; as such, this leads to about 25/29,000 to have a 2-3 label
     This will furhter be broken done into seperate datasets and thus lead to 25 of your 27 datasets (IF YOUR LUCKY) 
     classifers to be none. As such, they aren't even counted. 
     
     solution: don't include features that don't pass this step rule 
        1. they must be important or else your risking the problem
        2. a majoritity of the rows of this feature can not be NA or spaces 
                 
                 
    I think that since big datasets when they are cut down still have errors and get turned into none this is the problem. 
    As such, my found hypothesis is you need a large dataset --THIS IS THE SOLUTION I BELEIVE 
    Okay, problem figured out: when I add more variables for some reason all the classifers become none (WHY IS THIS IS THE SOLUTION) 
     '''