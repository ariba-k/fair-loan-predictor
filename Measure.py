import copy
import math
import sys
import numpy as np
from sklearn.metrics import confusion_matrix



def makeAddedColumns(test_df_copy, biased_version, sexCArray, raceCArray, ethnicityCArray, y_pred):
    arrayAddedColumns = []
    arrayPRIVAddedColumns = []
    arrayUNPRIVAddedColumns = []
    for i in range(len(biased_version)):
        test_df_copy['current_pred_'] = y_pred
        # test_df_copy["current_pred_"] = pd.to_numeric(test_df_copy.current_pred_, errors='coerce')

        # print(test_df_copy.dtypes)
        # print('Behind:',test_df_copy[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
        # test_df_copy['derived_sex'] = np.where((test_df_copy['derived_sex'] == 0), True, False)
        # print('Ahead:',test_df_copy[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))
        # print('Look: \n', np.where((test_df_copy['derived_sex'] == 0), True, False))
        test_df_copy['TP_' + biased_version[i]] = np.where((test_df_copy['action_taken'] == 1) &
                                                           (test_df_copy['current_pred_'] == 1) &
                                                           (test_df_copy['derived_sex'] == sexCArray[i]) & (test_df_copy['derived_race'] == raceCArray[i]) &
                                                           (test_df_copy['derived_ethnicity'] == ethnicityCArray[i]), 1, 0)
        arrayAddedColumns.append('TP_' + biased_version[i])

        # print(ethnicityCArray[i])
        # print(test_df_copy.head(20))

        test_df_copy['TN_' + biased_version[i]] = np.where((test_df_copy['action_taken'] == 0) &
                                                (test_df_copy['current_pred_'] == 0) &
                                                (test_df_copy['derived_sex'] == sexCArray[i]) & (test_df_copy['derived_race'] == raceCArray[i]) &
                                                (test_df_copy['derived_ethnicity'] == ethnicityCArray[i]), 1, 0)

        arrayAddedColumns.append('TN_' + biased_version[i])

        test_df_copy['FN_' + biased_version[i]] = np.where((test_df_copy['action_taken'] == 1) &
                                                (test_df_copy['current_pred_'] == 0) &
                                                (test_df_copy['derived_sex'] == sexCArray[i]) & (test_df_copy['derived_race'] == raceCArray[i]) &
                                                (test_df_copy['derived_ethnicity'] == ethnicityCArray[i]), 1, 0)

        arrayAddedColumns.append('FN_' + biased_version[i])

        test_df_copy['FP_' + biased_version[i]] = np.where((test_df_copy['action_taken'] == 0) &
                                                (test_df_copy['current_pred_'] == 1) &
                                                (test_df_copy['derived_sex'] == sexCArray[i]) & (test_df_copy['derived_race'] == raceCArray[i]) &
                                                (test_df_copy['derived_ethnicity'] == ethnicityCArray[i]), 1, 0)

        arrayAddedColumns.append('FP_' + biased_version[i])

        if (biased_version[i] == 'allBFNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allBFHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allBFJointEthnicitydataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allWFNHOLdataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allWFHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allWFJointEthnicitydataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointRaceFemaleNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointRaceFemaleHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointRaceFemaleJointEthnicitydataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allBMNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allBMHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allBMJointEthnicitydataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allWMNHOLdataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allWMHOLdataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allWMJointEthnicitydataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointRaceMaleNHOLdataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointRaceMaleHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointRaceMaleJointEthnicitydataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexBlacksNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexBlacksHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexBlacksJointEthnicitydataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexWhitesNHOLdataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexWhitesHOLdataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexWhitesJointEthnicitydataset'):
            arrayPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexJointRaceNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexJointRaceHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] == 'allJointSexJointRaceJointEthnicitydataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
    print(arrayAddedColumns)
    print(arrayUNPRIVAddedColumns)
    print(arrayPRIVAddedColumns)
    return arrayUNPRIVAddedColumns, arrayPRIVAddedColumns


def get_counts_editing(clf, x_train, y_train, x_test, y_test, test_df, biased_version, sexCArray, raceCArray, ethnicityCArray, metric):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    print(cnf_matrix)
    # print(classification_report(y_test, y_pred))


    TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

    test_df_copy = copy.deepcopy(test_df)
    fileToSaveTo1 = str(sys.path[0]) + '\\' + 'FirstTestDFCopy.csv'
    test_df_copy.to_csv(fileToSaveTo1)

    arrayUNPRIVAddedColumns, arrayPRIVAddedColumns = makeAddedColumns(test_df_copy, biased_version, sexCArray, raceCArray, ethnicityCArray, y_pred)

    fileToSaveTo = str(sys.path[0]) + '\\' + 'TestDFCopyAddedColumns.csv'
    test_df_copy.to_csv(fileToSaveTo)
    print(test_df_copy.head(200))
    print('After makeAddedColumns function:', len(test_df_copy))



    if metric == 'mEOD':
        return calculate_eod_other(test_df_copy, arrayUNPRIVAddedColumns, arrayPRIVAddedColumns)
    elif metric == 'mAOD':
        return calculate_aod_other(test_df_copy, arrayUNPRIVAddedColumns, arrayPRIVAddedColumns, calculate_eod(test_df_copy, arrayUNPRIVAddedColumns, arrayPRIVAddedColumns))
    elif metric == 'recall':
        return calculate_recall(TP, FP, FN, TN)
    elif metric == 'far':
        return calculate_far(TP, FP, FN, TN)
    elif metric == 'precision':
        return calculate_precision(TP, FP, FN, TN)
    elif metric == 'accuracy':
        return calculate_accuracy(TP, FP, FN, TN)


#----------------Calculates EOD Metric Function --------------

#Set up Arrays that will be used in the function to compute values


def calculate_eod_other(test_df_copy, arrayUNPRIV, arrayPRIV):
    arrayEOD1 = []
    arrayEOD2 = []
    arrayEOD3 = []
    arrayEOD4 = []

    print("Here is arrayUNPRIV:,", arrayUNPRIV)
    print("Here is arrayPRIV:", arrayPRIV)
    print(len(arrayUNPRIV))
    print(len(arrayPRIV))
    allTPRu = 0
    allTPRp = 0
    countu = 0
    countp = 0

    #This for loop will trek the arrayUnpriviliged function, which holds the TP, TN, FP, FN of all unprivileged datasets, and get the sum of the TP column as well as the FN column and put it in a list (e.g [TP, FN, TP, FN, TP, FN), where TP and FN are actual numerical values
    for k in range(len(arrayUNPRIV)):
        if k % 4 == 0 or k % 4 == 2:
           arrayEOD1.append(test_df_copy[arrayUNPRIV[k]].sum())
    print('EOD1:', arrayEOD1)
    print(len(arrayEOD1))

    for x in range((len(arrayEOD1) - 1)): # - 1 due to index error
        if x % 2 == 0:
            currentTP = arrayEOD1[x]
            currentFN = arrayEOD1[x+1]
            totalTPFN = arrayEOD1[x] + arrayEOD1[x + 1]
            if ((currentTP > 4 or currentFN > 4) and totalTPFN != 0):
                arrayEOD3.append(currentTP/totalTPFN)
    print('EOD3:', arrayEOD3)
    print(len(arrayEOD3))

    for l in range(len(arrayPRIV)):
        if l % 4 == 0 or l % 4 == 2:
           arrayEOD2.append(test_df_copy[arrayPRIV[l]].sum())
    print('EOD2:', arrayEOD2)
    print(len(arrayEOD2))

    for y in range((len(arrayEOD2) - 1)): # - 1 due to index error
        if y % 2 == 0:
            currentTP = arrayEOD2[y]
            currentFN = arrayEOD2[y + 1]
            totalTPFN = arrayEOD2[y] + arrayEOD2[y + 1]
            if ((currentTP > 4 or currentFN > 4) and totalTPFN != 0):
                arrayEOD4.append(currentTP / totalTPFN)
    print('EOD4:', arrayEOD4)
    print(len(arrayEOD4))

    for w in range(len(arrayEOD3)):
        allTPRu = allTPRu + arrayEOD3[w]
        countu = countu + 1
    avgTPRu = (allTPRu/countu)
    print(avgTPRu)

    for e in range(len(arrayEOD4)):
        allTPRp = allTPRp + arrayEOD4[e]
        countp = countp + 1
    avgTPRp = (allTPRp/countp)
    print(avgTPRp)

    EODdiff = avgTPRu - avgTPRp


    return EODdiff,

def calculate_eod(test_df_copy, arrayUNPRIV, arrayPRIV):
    arrayEOD1 = []
    arrayEOD2 = []
    arrayEOD3 = []
    arrayEOD4 = []

    allTPRu = 0
    allTPRp = 0
    countu = 0
    countp = 0

    #This for loop will trek the arrayUnpriviliged function, which holds the TP, TN, FP, FN of all unprivileged datasets, and get the sum of the TP column as well as the FN column and put it in a list (e.g [TP, FN, TP, FN, TP, FN), where TP and FN are actual numerical values
    for k in range(len(arrayUNPRIV)):
        if k % 4 == 0 or k % 4 == 2:
           arrayEOD1.append(test_df_copy[arrayUNPRIV[k]].sum())


    for x in range((len(arrayEOD1) - 1)): # - 1 due to index error
        if x % 2 == 0:
            currentTP = arrayEOD1[x]
            currentFN = arrayEOD1[x + 1]
            totalTPFN = arrayEOD1[x] + arrayEOD1[x + 1]
            if ((currentTP > 4 or currentFN > 4) and totalTPFN != 0):
                arrayEOD3.append(currentTP / totalTPFN)

    for l in range(len(arrayPRIV)):
        if l % 4 == 0 or l % 4 == 2:
           arrayEOD2.append(test_df_copy[arrayPRIV[l]].sum())

    for y in range((len(arrayEOD2) - 1)): # - 1 due to index error
        if y % 2 == 0:
            currentTP = arrayEOD2[y]
            currentFN = arrayEOD2[y + 1]
            totalTPFN = arrayEOD2[y] + arrayEOD2[y + 1]
            if ((currentTP > 4 or currentFN > 4) and totalTPFN != 0):
                arrayEOD4.append(currentTP / totalTPFN)

    for w in range(len(arrayEOD3)):
        allTPRu = allTPRu + arrayEOD3[w]
        countu = countu + 1
    avgTPRu = (allTPRu/countu)

    for e in range(len(arrayEOD4)):
        allTPRp = allTPRp + arrayEOD4[e]
        countp = countp + 1
    avgTPRp = (allTPRp/countp)

    EODdiff = avgTPRu - avgTPRp
    return EODdiff





#Calculates AOD Metric Function

def calculate_aod_other(test_df_copy, arrayUNPRIV, arrayPRIV, EODdiff):
    arrayAOD1 = []
    arrayAOD2 = []
    arrayAOD3 = []
    arrayAOD4 = []

    allFPRu = 0
    allFPRp = 0
    countu = 0
    countp = 0
    for k in range(len(arrayUNPRIV)):
        if k % 4 == 1 or k % 4 == 3:
            arrayAOD1.append(test_df_copy[arrayUNPRIV[k]].sum())
    print('AOD1:', arrayAOD1)
    print(len(arrayAOD1))

    for x in range((len(arrayAOD1) - 1)):  # - 1 due to index error
        if x % 2 == 0:
            currentFP = arrayAOD1[x + 1]
            totalFPTN = arrayAOD1[x] + arrayAOD1[x + 1]
            if (totalFPTN == 0):
                arrayAOD3.append(0)
            else:
                arrayAOD3.append(currentFP / totalFPTN)
    print('AOD3:', arrayAOD3)
    print(len(arrayAOD3))

    for l in range(len(arrayPRIV)):
        if l % 4 == 1 or l % 4 == 3:
            arrayAOD2.append(test_df_copy[arrayPRIV[l]].sum())
    print('AOD2:', arrayAOD2)
    print(len(arrayAOD2))

    for y in range((len(arrayAOD2) - 1)):  # - 1 due to index error
        if y % 2 == 0:
            currentFP = arrayAOD2[y + 1]
            totalFPTN = arrayAOD2[y] + arrayAOD2[y + 1]
            if (totalFPTN == 0):
                arrayAOD4.append(0)
            else:
                arrayAOD4.append(currentFP / totalFPTN)
    print('AOD4:', arrayAOD4)
    print(len(arrayAOD4))

    for w in range(len(arrayAOD3)):
        allFPRu = allFPRu + arrayAOD3[w]
        countu = countu + 1
    avgFPRu = (allFPRu / countu)

    for e in range(len(arrayAOD4)):
        allFPRp = allFPRp + arrayAOD4[e]
        countp = countp + 1
    avgFPRp = (allFPRp / countp)
    FPRdiff = (avgFPRu - avgFPRp)
    AODdiff = (FPRdiff + EODdiff) * 0.5
    return AODdiff


#Calculates Recall Metric
def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return recall

#Calculates Far Metric
def calculate_far(TP,FP,FN,TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return far

#Calculates Precision Metric
def calculate_precision(TP,FP,FN,TN):
    if (TP + FP) != 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    return prec

#Calculates Accuracy Metric
def calculate_accuracy(TP,FP,FN,TN):
    return (TP + TN)/(TP + TN + FP + FN)


#Starting/Base function that calls other functions depending upon the metric and paramaters
def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_version, sexCArray, raceCArray, ethnicityCArray, metric):
    df = copy.deepcopy(test_df)
    return get_counts_editing(clf, X_train, y_train, X_test, y_test, df, biased_version, sexCArray, raceCArray, ethnicityCArray, metric)

