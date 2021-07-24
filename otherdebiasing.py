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
    dataset_orig = pd.read_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\TestHMDA.csv',
                               dtype=object)
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
    dataset_orig = dataset_orig.drop(['census_tract', 'activity_year', 'lei', 'state_code', 'conforming_loan_limit',
    'derived_loan_product_type', 'derived_dwelling_category', 'loan_to_value_ratio', 'interest_rate',
    'rate_spread', 'total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points',
    'lender_credits', 'loan_term', 'prepayment_penalty_term', 'intro_rate_period', 'property_value',
    'multifamily_affordable_units', 'income', 'debt_to_income_ratio', 'applicant_ethnicity-2',
    'applicant_ethnicity-3', 'applicant_ethnicity-4', 'applicant_ethnicity-5', 'co-applicant_ethnicity-2',
    'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5', 'applicant_race-2',
    'applicant_race-3', 'applicant_race-4', 'applicant_race-5', 'co-applicant_race-2', 'co-applicant_race-3',
    'co-applicant_race-4', 'co-applicant_race-5', 'applicant_age_above_62', 'co-applicant_age_above_62', 'aus-2',
    'aus-3', 'aus-4', 'aus-5', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4', 'total_units',
    "applicant_age", "co-applicant_age"], axis=1)

    removeNA(list(dataset_orig.columns))
    removeExempt(list(dataset_orig.columns))
    removeBlank(list(dataset_orig.columns))
    dataset_orig.reset_index(drop=True, inplace=True)

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

    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
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
    D['derived_ethnicity'] = 0
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
    D['derived_race'] = 0
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
    D['derived_sex'] = 0
    # print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(30))
    numRows = len(D)
    X_train, y_train = D.loc[:, D.columns != 'action_taken'], D['action_taken']
    # --- LSR
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    return clf.fit(X_train, y_train)



##################################
# --------------------Classifiers ---------------------

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
    allBFNHOL_y = clf1.predict(row)
    allBFHOL_y = clf2.predict(row)
    allBFJointEthnicity_y = clf3.predict(row)
    allWFNHOL_y = clf4.predict(row)
    allWFHOL_y = clf5.predict(row)
    allWFJointEthnicity_y = clf6.predict(row)
    allJointRaceFemaleNHOL_y = clf7.predict(row)
    allJointRaceFemaleHOL_y = clf8.predict(row)
    allJointRaceFemaleJointEthnicity_y = clf9.predict(row)
    allBMNHOL_y = clf10.predict(row)
    allBMHOL_y = clf11.predict(row)
    allBMJointEthnicity_y = clf12.predict(row)
    allWMNHOL_y = clf13.predict(row)
    allWMHOL_y = clf14.predict(row)
    allWMJointEthnicity_y = clf15.predict(row)
    allJointRaceMaleNHOL_y = clf16.predict(row)
    allJointRaceMaleHOL_y = clf17.predict(row)
    allJointRaceMaleJointEthnicity_y = clf18.predict(row)
    allJointSexBlacksNHOL_y = clf19.predict(row)
    allJointSexBlacksHOL_y = clf20.predict(row)
    allJointSexBlacksJointEthnicity_y = clf21.predict(row)
    allJointSexWhitesNHOL_y = clf22.predict(row)
    allJointSexWhitesHOL_y = clf23.predict(row)
    allJointSexWhitesJointEthnicity_y = clf24.predict(row)
    allJointSexJointRaceNHOL_y = clf25.predict(row)
    allJointSexJointRaceHOL_y = clf26.predict(row)
    allJointSexJointRaceJointEthnicity_y = clf27.predict(row)

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
    if not ((arrayClassifiers[0][0] == arrayClassifiers[1][0]) and (arrayClassifiers[1][0] == arrayClassifiers[2][0])
            and (arrayClassifiers[2][0] == arrayClassifiers[3][0]) and (arrayClassifiers[3][0] == arrayClassifiers[4][0])
            and (arrayClassifiers[4][0] == arrayClassifiers[5][0]) and (arrayClassifiers[5][0] == arrayClassifiers[6][0])
            and (arrayClassifiers[6][0] == arrayClassifiers[7][0]) and (arrayClassifiers[7][0] == arrayClassifiers[8][0])
            and (arrayClassifiers[8][0] == arrayClassifiers[9][0]) and (arrayClassifiers[9][0] == arrayClassifiers[10][0])
            and (arrayClassifiers[10][0] == arrayClassifiers[11][0]) and (arrayClassifiers[11][0] == arrayClassifiers[12][0])
            and (arrayClassifiers[12][0] == arrayClassifiers[13][0]) and (arrayClassifiers[13][0] == arrayClassifiers[14][0])
            and (arrayClassifiers[14][0] == arrayClassifiers[15][0]) and (arrayClassifiers[15][0] == arrayClassifiers[16][0])
            and (arrayClassifiers[17][0] == arrayClassifiers[18][0]) and (arrayClassifiers[18][0] == arrayClassifiers[19][0])
            and (arrayClassifiers[19][0] == arrayClassifiers[20][0]) and (arrayClassifiers[20][0] == arrayClassifiers[21][0])
            and (arrayClassifiers[21][0] == arrayClassifiers[22][0]) and (arrayClassifiers[22][0] == arrayClassifiers[23][0])
            and (arrayClassifiers[23][0] == arrayClassifiers[24][0]) and (arrayClassifiers[24][0] == arrayClassifiers[25][0])
            and (arrayClassifiers[25][0] == arrayClassifiers[26][0])):
        dataset_orig = dataset_orig.drop(index)
#     else:
#         print(y_male_white[0],y_male_black[0],y_female_white[0],y_female_black[0])

print(dataset_orig.shape)


#for i in (len(arrayClassifiers)-1):
#   arrayConditional = []
#   if(arrayClassifiers[i][0] == None or arrayClassifiers[i+1][0]):
#
#   else:
#       arrayConditional.append(arrayClassifiers[i][0] == arrayClassifiers[i+1][0]))
#for m in (len(arrayConditional)):
#      if arrayConditional[m] == False:
#           dataset_orig = dataset.drop(index)
#
#
#