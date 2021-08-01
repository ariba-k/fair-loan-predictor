import copy
import math
import sys
import numpy as np
from sklearn.metrics import confusion_matrix



def get_counts(clf, x_train, y_train, x_test, y_test, test_df, biased_version, metric):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    print(cnf_matrix)
    # print(classification_report(y_test, y_pred))


    TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

    test_df_copy = copy.deepcopy(test_df)
    print('Before:', test_df_copy.head(20))

    test_df_copy['current_pred_'  + biased_version] = y_pred

    print('You\'re here', test_df_copy.head(20))

    test_df_copy['TP_' + biased_version + "_1"] = np.where((test_df_copy['action_taken'] == 1) &
                                           (test_df_copy['current_pred_' + biased_version] == 1) &
                                           (test_df_copy[biased_version] == 1), 1, 0)

    test_df_copy['TN_' + biased_version + "_1"] = np.where((test_df_copy['action_taken'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_version] == 0) &
                                                  (test_df_copy[biased_version] == 1), 1, 0)

    test_df_copy['FN_' + biased_version + "_1"] = np.where((test_df_copy['action_taken'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_version] == 0) &
                                                  (test_df_copy[biased_version] == 1), 1, 0)

    test_df_copy['FP_' + biased_version + "_1"] = np.where((test_df_copy['action_taken'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_version] == 1) &
                                                  (test_df_copy[biased_version] == 1), 1, 0)

    test_df_copy['TP_' + biased_version + "_0"] = np.where((test_df_copy['action_taken'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_version] == 1) &
                                                  (test_df_copy[biased_version] == 0), 1, 0)

    test_df_copy['TN_' + biased_version + "_0"] = np.where((test_df_copy['action_taken'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_version] == 0) &
                                                  (test_df_copy[biased_version] == 0), 1, 0)

    test_df_copy['FN_' + biased_version + "_0"] = np.where((test_df_copy['action_taken'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_version] == 0) &
                                                  (test_df_copy[biased_version] == 0), 1, 0)

    test_df_copy['FP_' + biased_version + "_0"] = np.where((test_df_copy['action_taken'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_version] == 1) &
                                                  (test_df_copy[biased_version] == 0), 1, 0)

    a = test_df_copy['TP_' + biased_version + "_1"].sum()
    b = test_df_copy['TN_' + biased_version + "_1"].sum()
    c = test_df_copy['FN_' + biased_version + "_1"].sum()
    d = test_df_copy['FP_' + biased_version + "_1"].sum()
    e = test_df_copy['TP_' + biased_version + "_0"].sum()
    f = test_df_copy['TN_' + biased_version + "_0"].sum()
    g = test_df_copy['FN_' + biased_version + "_0"].sum()
    h = test_df_copy['FP_' + biased_version + "_0"].sum()

    if metric == 'aod':
        return calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric == 'eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    elif metric == 'recall':
        return calculate_recall(TP, FP, FN, TN)
    elif metric == 'far':
        return calculate_far(TP, FP, FN, TN)
    elif metric == 'precision':
        return calculate_precision(TP, FP, FN, TN)
    elif metric == 'accuracy':
        return calculate_accuracy(TP, FP, FN, TN)


arrayAddedColumns = []
arrayPRIVAddedColumns = []
arrayUNPRIVAddedColumns = []
def makeAddedColumns(test_df_copy, biased_version, sexCArray, raceCArray, ethnicityCArray, y_pred):
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

        if(biased_version[i] == 'allBFNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif(biased_version[i] == 'allBFHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif(biased_version[i] == 'allBFJointEthnicitydataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif(biased_version[i] == 'allWFNHOLdataset'):
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
        elif (biased_version[i] ==  'allBMNHOLdataset'):
            arrayUNPRIVAddedColumns.append('TP_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('TN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FN_' + biased_version[i])
            arrayUNPRIVAddedColumns.append('FP_' + biased_version[i])
        elif (biased_version[i] ==  'allBMHOLdataset'):
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

    makeAddedColumns(test_df_copy, biased_version, sexCArray, raceCArray, ethnicityCArray, y_pred)

    fileToSaveTo = str(sys.path[0]) + '\\' + 'TestDFCopyAddedColumns.csv'
    test_df_copy.to_csv(fileToSaveTo)
    print(test_df_copy.head(200))
    print('After makeAddedColumns function:', len(test_df_copy))

    print('EOD:', calculate_eod_other(test_df_copy, arrayUNPRIVAddedColumns, arrayPRIVAddedColumns))

    #
    # if metric == 'aod':
    # # return calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    # elif metric == 'eod':
    # # return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    # elif metric == 'recall':
    #     return calculate_recall(TP, FP, FN, TN)
    # elif metric == 'far':
    #     return calculate_far(TP, FP, FN, TN)
    # elif metric == 'precision':
    #     return calculate_precision(TP, FP, FN, TN)
    # elif metric == 'accuracy':
    #     return calculate_accuracy(TP, FP, FN, TN)

arrayEOD1 = []
arrayEOD2 = []
arrayEOD3 = []
arrayEOD4 = []

def calculate_eod_other(test_df_copy, arrayUNPRIV, arrayPRIV):
    allTPRu = 0
    allTPRp = 0
    countu = 0
    countp = 0
    for k in range(len(arrayUNPRIV)):
        if k % 4 == 0 or k % 4 == 2:
           arrayEOD1.append(test_df_copy[arrayUNPRIV[k]].sum()) #The problem is most likely originating here and is due to the; the sum function might not be the problem and it is actually the fact that you do the makeAddColumns function and creation of columns incorrectly, which leads to the sum actually being 0
    print('EOD1:', arrayEOD1)

    for x in range((len(arrayEOD1) - 1)): # - 1 due to index error
        if x % 2 == 0:
            currentTP = arrayEOD1[x]
            totalTPFP = arrayEOD1[x] + arrayEOD1[x + 1]
            if (totalTPFP == 0):
                arrayEOD3.append(0)
            else:
                arrayEOD3.append(currentTP/totalTPFP)
    print('EOD3:', arrayEOD3)

    for l in range(len(arrayPRIV)):
        if l % 4 == 0 or l % 4 == 2:
           arrayEOD2.append(test_df_copy[arrayPRIV[l]].sum())
    print('EOD2:', arrayEOD2)

    for y in range((len(arrayEOD2) - 1)): # - 1 due to index error
        if y % 2 == 0:
            currentTP = arrayEOD2[y]
            totalTPFP = arrayEOD2[y] + arrayEOD2[y + 1]
            if (totalTPFP == 0):
                arrayEOD4.append(0)
            else:
                arrayEOD4.append(currentTP/totalTPFP)
    print('EOD4:', arrayEOD4)

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

arrayAOD1 = []
arrayAOD2 = []
arrayAOD3 = []
arrayAOD4 = []
def calculate_aod_other(test_df_copy, arrayUNPRIV, arrayPRIV, EODdiff):
    allTPRu = 0
    allTPRp = 0
    countu = 0
    countp = 0
    for k in range(len(arrayUNPRIV)):
        if k % 4 == 0 or k % 4 == 3:
            arrayEOD1.append(test_df_copy[arrayUNPRIV[k]].sum())  # The problem is most likely originating here and is due to the; the sum function might not be the problem and it is actually the fact that you do the makeAddColumns function and creation of columns incorrectly, which leads to the sum actually being 0
    print('EOD1:', arrayEOD1)

    for x in range((len(arrayEOD1) - 1)):  # - 1 due to index error
        if x % 2 == 0:
            currentTP = arrayEOD1[x]
            totalTPFP = arrayEOD1[x] + arrayEOD1[x + 1]
            if (totalTPFP == 0):
                arrayEOD3.append(0)
            else:
                arrayEOD3.append(currentTP / totalTPFP)
    print('EOD3:', arrayEOD3)

    for l in range(len(arrayPRIV)):
        if l % 4 == 0 or l % 4 == 3:
            arrayEOD2.append(test_df_copy[arrayPRIV[l]].sum())
    print('EOD2:', arrayEOD2)

    for y in range((len(arrayEOD2) - 1)):  # - 1 due to index error
        if y % 2 == 0:
            currentTP = arrayEOD2[y]
            totalTPFP = arrayEOD2[y] + arrayEOD2[y + 1]
            if (totalTPFP == 0):
                arrayEOD4.append(0)
            else:
                arrayEOD4.append(currentTP / totalTPFP)
    print('EOD4:', arrayEOD4)

    for w in range(len(arrayEOD3)):
        allTPRu = allTPRu + arrayEOD3[w]
        countu = countu + 1
    avgTPRu = (allTPRu / countu)

    for e in range(len(arrayEOD4)):
        allTPRp = allTPRp + arrayEOD4[e]
        countp = countp + 1
    avgTPRp = (allTPRp / countp)

    EODdiff = avgTPRu - avgTPRp
    return EODdiff


def calculate_average_odds_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
	TPR_male = TP_male/(TP_male+FN_male)
	TPR_female = TP_female/(TP_female+FN_female)
	FPR_male = FP_male/(FP_male+TN_male)
	FPR_female = FP_female/(FP_female+TN_female)
	average_odds_difference = abs(abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))/2
	#print("average_odds_difference",average_odds_difference)
	return average_odds_difference


def calculate_equal_opportunity_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
	TPR_male = TP_male/(TP_male+FN_male)
	TPR_female = TP_female/(TP_female+FN_female)
	equal_opportunity_difference = abs(TPR_male - TPR_female)
	#print("equal_opportunity_difference:",equal_opportunity_difference)
	return equal_opportunity_difference



def calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    TPR_male = TP_male/(TP_male+FN_male)
    TPR_female = TP_female/(TP_female+FN_female)
    print("TPR_male:",TPR_male,"TPR_female:",TPR_female)
    diff = (TPR_male - TPR_female)
    return round(diff,2)

def calculate_FPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    FPR_male = FP_male/(FP_male+TN_male)
    FPR_female = FP_female/(FP_female+TN_female)
    print("FPR_male:",FPR_male,"FPR_female:",FPR_female)
    diff = (FPR_female - FPR_male)
    return round(diff,2)


def calculate_imbalance(full_df):
    full_df_copy = copy.deepcopy(full_df)

def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return recall

def calculate_far(TP,FP,FN,TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return far

def calculate_precision(TP,FP,FN,TN):
    if (TP + FP) != 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    return prec

def calculate_accuracy(TP,FP,FN,TN):
    return (TP + TN)/(TP + TN + FP + FN)


def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_version, sexCArray, raceCArray, ethnicityCArray, metric):
    df = copy.deepcopy(test_df)
    return get_counts_editing(clf, X_train, y_train, X_test, y_test, df, biased_version, sexCArray, raceCArray, ethnicityCArray, metric)

