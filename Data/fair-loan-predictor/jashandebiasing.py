## Here I am dividing the data first based onto protected attribute value and then train two separate models
import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath('..'))
fileloc = str(sys.path[0]) + '\\Data\\' + 'BalancedKYHMDA.csv'

##----KEY FUNCTIONS----##
# ==========================ABOVE IMPORTS========================================
#:Training dataset D, Sensitive attribute S, Binary
# classification model M trained on D, Input space
# similarity threshold delta
def resetDataset():
    dataset_orig = pd.read_csv(fileloc, dtype=object)
    print(dataset_orig.shape)

#####------------------Scaling?------------------------------------
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
####--------------------End of Scaling-----------------------------


    # print(dataset_orig[['derived_msa-md','derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
    # divide the data based on sex
    # dataset_new = dataset_orig.groupby(dataset_orig['derived_sex'] == 0)
    # print(dataset_new[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(20))
    fileToSaveTo1 =  str(sys.path[0]) + '\\' + 'AfterScalingTestHMDA.csv'
    dataset_orig.to_csv(fileToSaveTo1)

    return dataset_orig




#===============================================DONE WITH PREPROCESSING =======================================================
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

transitioner = 1

if (transitioner == 1):
    x1 = 0
    x2 =.5
    x3 = 1
else:
    x1 = 0
    x2 = 1
    x3 = 2



# =============================FIRST LAYER DIVIDE=====================================
allFemaleDataset = splittingDataset('derived_sex', [x2, x3])
print('Printing all female \n',
      allFemaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(55))
# allFemaleDataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\FDebuggingScaling.csv')


def allFemaleReset():
    allFemaleDataset = splittingDataset('derived_sex', [x2, x3])
    return allFemaleDataset


allMaleDataset = splittingDataset('derived_sex', [x1, x3])
# print('Printing all male \n',
#       allMaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allMaleReset():
    allMaleDataset = splittingDataset('derived_sex', [x1, x3])
    return allMaleDataset


allJointSexDataset = splittingDataset('derived_sex', [x1, x2])
# print('Printing all Joint \n',
#       allJointSexDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allJointSexReset():
    allJointSexDataset = splittingDataset('derived_sex', [x1, x2])
    return allJointSexDataset


# ===============================Second Layer Divide====================================
# First White and Black Females
allBFdataset = splittingDatasetSecondLayer('derived_race',[x2, x3], allFemaleDataset)
print('Printing all BFs \n',
      allBFdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\BFDebuggingScaling.csv')
def allBFReset():
    allFemaleDataset = allFemaleReset()
    allBFdataset = splittingDatasetSecondLayer('derived_race',[x2, x3], allFemaleDataset)
    return allBFdataset


allFemaleDataset = allFemaleReset()
'''OKAY, ARASH ABOVE IS THE MAIN CHANGE; ESSENTIALLY, YOU HAVE TO RESET THE DATASET, WHICHEVER ONE YOU ARE USING,
BEFORE YOU USE IT FOR ANOTHER SPLIT; THIS IS WHY I DO THE BF SPLIT AND THEN RESET FOR THE WF SPLIT. I DO THIS BY
MAKING A FUNCTION CALLED allFemaleReset(). FYI, we are going to have to make a lot of functions, but I found
this is the most efficent method to use'''
allWFdataset = splittingDatasetSecondLayer('derived_race',[x1, x3], allFemaleDataset)
# print('Printing all WFs \n',
#       allWFdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allWFReset():
    allFemaleDataset = allFemaleReset()
    allWFdataset = splittingDatasetSecondLayer('derived_race',[x1, x3], allFemaleDataset)
    return allWFdataset


allFemaleDataset = allFemaleReset()
allJointRaceFemaleDataset = splittingDatasetSecondLayer('derived_race', [x1, x2], allFemaleDataset)
# print('Printing all JointRaceFemales \n',
#       allJointRaceFemaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allJointRaceFemaleReset():
    allFemaleDataset = allFemaleReset()
    allJointRaceFemaleDataset = splittingDatasetSecondLayer('derived_race', [x1, x2], allFemaleDataset)
    return allJointRaceFemaleDataset


# ---------------------------second white, black males--------------------------------------------------------------------------------------------
allBMdataset = splittingDatasetSecondLayer('derived_race',[x2, x3], allMaleDataset)
# print('Printing all BMs \n',
#       allBMdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allBMReset():
    allMaleDataset = allMaleReset()
    allBMdataset = splittingDatasetSecondLayer('derived_race',[x2, x3], allMaleDataset)
    return allBMdataset


allMaleDataset = allMaleReset()
allWMdataset = splittingDatasetSecondLayer('derived_race',[x1, x3], allMaleDataset)
# print('Printing all WMs \n',
#       allWMdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allWMReset():
    allMaleDataset = allMaleReset()
    allWMdataset = splittingDatasetSecondLayer('derived_race',[x1, x3], allMaleDataset)
    return allWMdataset


allMaleDataset = allMaleReset()
allJointRaceMaleDataset = splittingDatasetSecondLayer('derived_race', [x1, x2], allMaleDataset)
# print('Printing all JointRaceMales \n',
#       allJointRaceMaleDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


def allJointRaceMaleReset():
    allMaleDataset = allMaleReset()
    allJointRaceMaleDataset = splittingDatasetSecondLayer('derived_race', [x1, x2], allMaleDataset)
    return allJointRaceMaleDataset


# -------------------Third Joint Sex Races -----------------
allJointSexBlacksDataset = splittingDatasetSecondLayer('derived_race',[x2, x3], allJointSexDataset)
# print('Printing all JointSexBlacks \n',
#       allJointSexBlacksDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allJointSexBlacksReset():
    allJointSexDataset = allJointSexReset()
    allJointSexBlacksDataset = splittingDatasetSecondLayer('derived_race',[x2, x3], allJointSexDataset)
    return allJointSexBlacksDataset


allJointSexDataset = allJointSexReset()
allJointSexWhitesDataset = splittingDatasetSecondLayer('derived_race',[x1, x3], allJointSexDataset)
# print('Printing all JointSexWhites \n',
#       allJointSexWhitesDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allJointSexWhitesReset():
    allJointSexDataset = allJointSexReset()
    allJointSexWhitesDataset = splittingDatasetSecondLayer('derived_race',[x1, x3], allJointSexDataset)
    return allJointSexWhitesDataset


allJointSexDataset = allJointSexReset()
allJointSexJointRaceDataset = splittingDatasetSecondLayer('derived_race', [x1, x2], allJointSexDataset)
# print('Printing all JointSex and JointRaces \n',
#       allJointSexJointRaceDataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))


# allBFdataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\allBFDataset.csv')
def allJointSexJointRaceReset():
    allJointSexDataset = allJointSexReset()
    allJointSexJointRaceDataset = splittingDatasetSecondLayer('derived_race', [x1, x2], allJointSexDataset)
    return allJointSexJointRaceDataset


# ===============================Third Layer Divide====================================
# ---------------1. BF Ethnic Split --------------
allBFHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allBFdataset)
# print('Printing all BFHOL \n',
#       allBFHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBFdataset = allBFReset()
allBFNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allBFdataset)
print('Printing all BFNHOLs \n',
      allBFNHOLdataset[['loan_amount', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
print('Num of Rows: \n', (len(allBFNHOLdataset)))
# allBFNHOLdataset.to_csv(r'C:\Users\Arash\OneDrive\Documents\GitHub\fair-loan-predictor\NewBFNHOLDebuggingSCALING.csv')

allBFdataset = allBFReset()
allBFJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2], allBFdataset)
# print('Printing all BFJointEthnicitys \n',
#       allBFJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------2. WF Ethnic Split --------------
allWFHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allWFdataset)
# print('Printing all WFHOL \n',
#       allWFHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWFdataset = allWFReset()
allWFNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allWFdataset)
# print('Printing all WFNHOLs \n',
#       allWFNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWFdataset = allWFReset()
allWFJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2], allWFdataset)
# print('Printing all WFJointEthnicitys \n',
#       allWFJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------3. JointRaceFemale Ethnic Split --------------
allJointRaceFemaleHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allJointRaceFemaleDataset)
# print('Printing all JointRaceFemaleNHOL \n',
#       allJointRaceFemaleHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

allJointRaceFemaleDataset = allJointRaceFemaleReset()
allJointRaceFemaleNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allJointRaceFemaleDataset)
# print('Printing all JointRaceFemaleHOL \n',
#       allJointRaceFemaleNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointRaceFemaleDataset = allJointRaceFemaleReset()
allJointRaceFemaleJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2],
                                                                      allJointRaceFemaleDataset)
# print('Printing all JointRaceFemaleJointEthnicity \n', allJointRaceFemaleJointEthnicitydataset[
#     ['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------4. BM Ethnic Split --------------
allBMHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allBMdataset)
# print('Printing all BMHOL \n',
#       allBMHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBMdataset = allBMReset()
allBMNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allBMdataset)
# print('Printing all BMNHOLs \n',
#       allBMNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allBMdataset = allBMReset()
allBMJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2], allBMdataset)
# print('Printing all BMJointEthnicitys \n',
#       allBMJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------5. WM Ethnic Split --------------
allWMHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allWMdataset)
# print('Printing all WMHOL \n',
#       allWMHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWMdataset = allWMReset()
allWMNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allWMdataset)
# print('Printing all WMNHOL \n',
#       allWMNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allWMdataset = allWMReset()
allWMJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2], allWMdataset)
# print('Printing all WMJointEthnicity \n',
#       allWMJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
# ---------------6. JointRaceMale Ethnic Split --------------
allJointRaceMaleHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allJointRaceMaleDataset)
# print('Printing all JointRaceMaleHOL \n',
#       allJointRaceMaleHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointRaceMaleDataset = allJointRaceMaleReset()
allJointRaceMaleNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allJointRaceMaleDataset)
# print('Printing all JointRaceMaleNHOL \n',
#       allJointRaceMaleNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointRaceMaleDataset = allJointRaceMaleReset()
allJointRaceMaleJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2],
                                                                    allJointRaceMaleDataset)
# print('Printing all JointRaceMaleJointEthnicity \n',
#       allJointRaceMaleJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(
#           50))
# ---------------7. JointSexBlacks Ethnic Split------------------
allJointSexBlacksHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allJointSexBlacksDataset)
# print('Printing all JointSexBlacksHOL \n',
#       allJointSexBlacksHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexBlacksDataset = allJointSexBlacksReset()
allJointSexBlacksNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allJointSexBlacksDataset)
# print('Printing all JointSexBlacksNHOL \n',
#       allJointSexBlacksNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexBlacksDataset = allJointSexBlacksReset()
allJointSexBlacksJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2],
                                                                     allJointSexBlacksDataset)
# print('Printing all JointSexBlacksJointEthnicity \n',
#       allJointSexBlacksJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(
#           50))
# ---------------8. JointSexWhites Ethnic Split------------------
allJointSexWhitesHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allJointSexWhitesDataset)
# print('Printing all JointSexWhitesHOL \n',
#       allJointSexWhitesHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexWhitesDataset = allJointSexWhitesReset()
allJointSexWhitesNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allJointSexWhitesDataset)
# print('Printing all JointSexWhitesNHOL \n',
#       allJointSexWhitesNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexWhitesDataset = allJointSexWhitesReset()
allJointSexWhitesJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2],
                                                                     allJointSexWhitesDataset)
# print('Printing all JointSexWhitesJointEthnicity \n',
#       allJointSexWhitesJointEthnicitydataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(
#           50))
# ---------------9. JointSexJointRace Ethnic Split------------------
allJointSexJointRaceHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x2, x3], allJointSexJointRaceDataset)
# print('Printing all JointSexJointRaceHOL \n',
#       allJointSexJointRaceHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexJointRaceDataset = allJointSexJointRaceReset()
allJointSexJointRaceNHOLdataset = splittingDatasetSecondLayer('derived_ethnicity',[x1, x3], allJointSexJointRaceDataset)
# print('Printing all JointSexJointRaceNHOL \n',
#       allJointSexJointRaceNHOLdataset[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
allJointSexJointRaceDataset = allJointSexJointRaceReset()
allJointSexJointRaceJointEthnicitydataset = splittingDatasetSecondLayer('derived_ethnicity', [x1, x2],
                                                                        allJointSexJointRaceDataset)
# print('Printing all JointSexJointRaceJointEthnicity \n', allJointSexJointRaceJointEthnicitydataset[
#     ['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
fileToSaveTo2 = str(sys.path[0]) + '\\' + 'allJointSexJointRaceJointEthnicitydataset.csv'
allJointSexJointRaceJointEthnicitydataset.to_csv(fileToSaveTo2)



arrayDatasets = [
                allBFNHOLdataset,
                allBFHOLdataset,
                allBFJointEthnicitydataset,
                allWFNHOLdataset,
                allWFHOLdataset,
                allWFJointEthnicitydataset,
                allJointRaceFemaleNHOLdataset,
                allJointRaceFemaleHOLdataset,
                allJointRaceFemaleJointEthnicitydataset,
                allBMNHOLdataset,
                allBMHOLdataset,
                allBMJointEthnicitydataset,
                allWMNHOLdataset,
                allWMHOLdataset,
                allWMJointEthnicitydataset,
                allJointRaceMaleNHOLdataset,
                allJointRaceMaleHOLdataset,
                allJointRaceMaleJointEthnicitydataset,
                allJointSexBlacksNHOLdataset,
                allJointSexBlacksHOLdataset,
                allJointSexBlacksJointEthnicitydataset,
                allJointSexWhitesNHOLdataset,
                allJointSexWhitesHOLdataset,
                allJointSexWhitesJointEthnicitydataset,
                allJointSexJointRaceNHOLdataset,
                allJointSexJointRaceHOLdataset,
                allJointSexJointRaceJointEthnicitydataset
                 ]

#
# def scaleDatasets():
#     for i in range(len(arrayDatasets)):
#         try:
#             scaler = MinMaxScaler()
#             arrayDatasets[i] = pd.DataFrame(scaler.fit_transform(arrayDatasets[i]), columns=arrayDatasets[i].columns)
#         except:
#             x = 1
#
# scaleDatasets()


##################################
# Classifier Function (helps to build classifiers):

def createClassifier(D):
    D.reset_index(drop=True, inplace=True)
    print(D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

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
    one_count = 0
    zero_count = 0
    for x in range(numRows):
      action_element = D.loc[x].iat[numCols]
      if(action_element == 0 or action_element == 0.0):
          hasZero = True
      if(action_element == 1 or action_element == 1.0):
          hasOne = True

    print('this is one count:', one_count)
    print('this is zero count:', zero_count)

    #see which var needs to be balanced
    if(zero_count > one_count):
        threshold = zero_count - one_count
        row_to_take_out = 0
    elif(one_count > zero_count):
        threshold = one_count - zero_count
        row_to_take_out = 1
    else:
        threshold = 0

    numDeleted = 0
    #start balancing
    while (numDeleted < threshold):
        numRows = len(D) - 1
        print('this is threshold', threshold)
        numRandom = random.randint(0, numRows)
        # print(numRandom)
        randomRowActionTaken = D.loc[numRandom].iat[numCols]
        print(randomRowActionTaken)
        if (randomRowActionTaken == row_to_take_out):
            D = D.drop(numRandom)
            D.reset_index(drop=True, inplace=True)
            numDeleted = numDeleted + 1
            print('NumDeleted Val:', numDeleted)

    print('-------------after balancing---------:', D[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
    numRows = len(D)
    for x in range(numRows):
      action_element = D.loc[x].iat[numCols]
      if(action_element == 0 or action_element == 0.0):
          hasZero = True
      if(action_element == 1 or action_element == 1.0):
          hasOne = True

    X_train, y_train = D.loc[:, D.columns != 'action_taken'], D['action_taken']
    # --- LSR
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=200)
    print(hasZero)
    print(hasOne)
    print(numRows >= 2)


    if numRows >= 40 and hasZero and hasOne:
        return clf.fit(X_train, y_train)
    else:
        return None

##################################
# --------------------Classifiers ---------------------
# #THIS IS OUT OF ORDER -- PROBLEM?????
# allFemaleDataset = allFemaleReset()
# clf0 = createClassifier(allFemaleDataset)
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
numOfOnes0 = 0
numOfZeros0 = 0
numOfNones0 = 0
numOfOnes1 = 0
numOfZeros1 = 0
numOfOnes2 = 0
numOfZeros2 = 0
for index, row in dataset_orig.iterrows():
    y_label = [row.values[len(row.values) - 1]]
    row = [row.values[0:len(row.values) - 1]]
    # try:
    #     allFemale_y = clf0.predict(row)
    #     if (allFemale_y[0] == 1):
    #         numOfOnes0 = numOfOnes0 + 1
    #     else:
    #         numOfZeros0 = numOfZeros0 + 1
    # except:
    #     allFemale_y = None
    #     numOfNones0 = numOfNones0 + 1
    try:
        allBFNHOL_y = clf1.predict(row)
        if (allBFNHOL_y[0] == 1):
            numOfOnes1 = numOfOnes1 + 1
        else:
            numOfZeros1 = numOfZeros1 + 1
    except:
        allBFNHOL_y = None
    try:
        allBFHOL_y = clf2.predict(row)
        if (allBFHOL_y[0] == 1):
            numOfOnes2 = numOfOnes2 + 1
        else:
            numOfZeros2 = numOfZeros2 + 1
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

    print(row)
    print('printing y', y_label)
    print(allBFNHOL_y, allBFHOL_y,
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
                        allJointSexJointRaceJointEthnicity_y)





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


    # for i in range((len(arrayClassifiers) - 1)):
    #     arrayConditional = []
    #     if arrayClassifiers[i] != None and arrayClassifiers[i + 1] != None:
    #         if not(arrayClassifiers[i][0] == arrayClassifiers[i + 1][0]):
    #             try:
    #              dataset_orig = dataset_orig.drop(index)
    #             except:
    #              jashan = "yes"
    #              # print('Yeah, already deleted')
    arrayConditional = []
    real_array_conditonal = []
    for i in range((len(arrayClassifiers) - 1)):
        temp = arrayClassifiers[i]
        if(temp != None):
            for j in range((len(arrayClassifiers) - 1)):
                    if(arrayClassifiers[j] != None):
                        result = (temp[0] == arrayClassifiers[j][0])
                        arrayConditional.append(result)
            result2 = arrayConditional.count(arrayConditional[0]) == len(arrayConditional)
            real_array_conditonal.append(result2)

    print('real array conditonal:', real_array_conditonal)
    for m in range((len(real_array_conditonal))):
        if (real_array_conditonal[m] == False):
            try:
              dataset_orig = dataset_orig.drop(index)
            except:
                jashan = "yes"

print(dataset_orig.shape)
print(list(dataset_orig.columns))
print(dataset_orig[['derived_ethnicity', 'derived_race', 'derived_sex','action_taken']].head(50))
# print(numOfOnes0)
# print(numOfZeros0)
# print("Nones for female:", numOfNones0)
# print(numOfOnes1)
# print(numOfZeros1)
# print(numOfOnes2)
# print(numOfZeros2)

fileToSaveTo3 = str(sys.path[0]) + '\\Data\\'+ 'KYDebiasedDataset.csv'
dataset_orig.to_csv(fileToSaveTo3)


#
#
#
# ################################################################################################################################
# ################################################################################################################################
# print(dataset_orig.shape)
# np.random.seed(0)
# ## Divide into train,validation,test
# dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=0,shuffle = True)
#
# X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'action_taken'], dataset_orig_train['action_taken']
# X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']
#
# # --- LSR
# clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# # --- CART
# # clf = tree.DecisionTreeClassifier()
#
# # clf.fit(X_train, y_train)
# # import matplotlib.pyplot as plt
# # y = np.arange(len(dataset_orig_train.columns)-1)
# # plt.barh(y,clf.coef_[0])
# # plt.yticks(y,dataset_orig_train.columns)
# # plt.show()
#
# # print(clf_male.coef_[0])
# # y_pred = clf.predict(X_test)
# # cnf_matrix_test = confusion_matrix(y_test,y_pred)
#
# # print(cnf_matrix_test)
# # TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()
#
#
# print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'recall'))
# print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'far'))
# print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'precision'))
# print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'accuracy'))
# print("aod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'aod'))
# print("eod sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_sex', 'eod'))
#
# print("TPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_race', 'TPR'))
# print("FPR:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'derived_race', 'FPR'))
