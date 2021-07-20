from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from random import randint

import pandas as pd


##----KEY FUNCTIONS----##


# ==========================ABOVE IMPORTS========================================
#:Training dataset D, Sensitive attribute S, Binary
# classification model M trained on D, Input space
# similarity threshold delta


HMDA_df = pd.read_csv(r'C:\Users\Arash\PycharmProjects\Debiasing\TestHMDA.csv', dtype=object)
###--------------------Sex------------------------
indexNames1 = HMDA_df[HMDA_df['derived_sex'] == "Sex Not Available"].index
HMDA_df.drop(indexNames1 , inplace=True)
###-------------------Races-----------------------
indexNames2 = HMDA_df[HMDA_df['derived_race'] == "American Indian or Alaska Native"].index
HMDA_df.drop(indexNames2 , inplace=True)
indexNames3 = HMDA_df[HMDA_df['derived_race'] == "Native Hawaiian or Other Pacific Islander"].index
HMDA_df.drop(indexNames3 , inplace=True)
indexNames4 = HMDA_df[HMDA_df['derived_race'] == "2 or more minority races"].index
HMDA_df.drop(indexNames4 , inplace=True)
indexNames5 = HMDA_df[HMDA_df['derived_race'] == "Asian"].index
HMDA_df.drop(indexNames5 , inplace=True)
indexNames6 = HMDA_df[HMDA_df['derived_race'] == "Free Form Text Only" ].index
HMDA_df.drop(indexNames6 , inplace=True)
indexNames7 = HMDA_df[HMDA_df['derived_race'] == "Race Not Available"].index
HMDA_df.drop(indexNames7 , inplace=True)
####----------------Ethnicity-------------------
indexNames8 = HMDA_df[HMDA_df['derived_ethnicity'] == "Ethnicity Not Available"].index
HMDA_df.drop(indexNames8 , inplace=True)
indexNames9 = HMDA_df[HMDA_df['derived_ethnicity'] == "Free Form Text Only"].index
HMDA_df.drop(indexNames9 , inplace=True)
####---------------Reset Indexes----------------
HMDA_df.reset_index(drop=True, inplace=True)


HMDA_df.loc[(HMDA_df.derived_sex == 'Joint'),'derived_sex']='Joint3'
HMDA_df.loc[(HMDA_df.derived_race == 'Joint'),'derived_race']='Joint2'
HMDA_df.loc[(HMDA_df.derived_ethnicity == 'Joint'),'derived_ethnicity']='Joint1'

###----------------Begin Code------------------
print(HMDA_df[['derived_ethnicity', 'derived_race', 'derived_sex']].head(70))





# A function created used to calculate the euclidian distance between two points
# def euclidianDistance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2)) ** 2)
# #

# Randomly samples the data to create a random individual
def Sample(D):
    # D is the dataset that we are using--in other words, HMDA_df
    numRows = len(D) - 1
    numCols = len(D.columns) - 1
    array = []
    for i in range(numCols):
        numRandom = randint(0, numRows)
        currentRandFeature = D.loc[numRandom].iat[i]
        array.append(currentRandFeature)
    return array


# Generates a similar individual to the sample individual; same person but with different sensitive params.
def SimilarIndividuals(D, A1, S, Lambda):
    ethnicities = D[S[0]].unique()
    print(ethnicities)
    races = D[S[1]].unique()
    print(races)
    sexes = D[S[2]].unique()
    print(sexes)
    simIndividuals_r2 = []

    for m in range(26):
        for i in range(len(A1)):
            if (A1[i] == ethnicities[0] or A1[i] == ethnicities[1] or A1[i] == ethnicities[2]):
                if (A1[i] == ethnicities[0]):
                    startNum1 = 0
                if (A1[i] == ethnicities[1]):
                    startNum1 = 1
                if (A1[i] == ethnicities[2]):
                    startNum1 = 2
                startNum1 = (startNum1 + 1) % 3
                A1[i] = ethnicities[startNum1]
            if (A1[i] == races[0] or A1[i] == races[1] or A1[i] == races[2]):
                if (A1[i] == races[0]):
                    startNum2 = 0
                if (A1[i] == races[1]):
                    startNum2 = 1
                if (A1[i] == races[2]):
                    startNum2 = 2
                startNum2 = (startNum2 + 1) % 3
                A1[i] = races[startNum2]
            if (A1[i] == sexes[0] or A1[i] == sexes[1] or A1[i] == sexes[2]):
                if (A1[i] == sexes[0]):
                    startNum3 = 0
                if (A1[i] == sexes[1]):
                    startNum3 = 1
                if (A1[i] == sexes[2]):
                    startNum3 = 2
                startNum3 = (startNum3 + 1) % 3
                A1[i] = sexes[startNum3]
        simIndividuals_r2.append(A1)
    return simIndividuals_r2


    # I started to write code

    #
    # for k in range(len(S)):
    #     uniqueVal = D[S[k]].unique()
    #     for l in range(len(uniqueVal)):
    #         if(A1[i] == uniqueVal[l]):
    #             currentUnique = uniqueVal[l]
    #             while(uniqueVal[randint(0, len(uniqueVal))] == currentUnique):
    #                 forcedOtherString = uniqueVal[randint(0, len(uniqueVal))]
    #                 A1[i] = forcedOtherString



tempA1 = Sample(HMDA_df)
print(tempA1)
print(SimilarIndividuals(HMDA_df, tempA1, ["derived_ethnicity", "derived_race", "derived_sex"], 0))
#
# def sortDataset(D, S, M, delta):
# This function will generate similar pairs of indivuals for each training sample in a dataset

# def generateSimilarPairs(D, S, delta):
#     similarPairs = np.array()
#     loopLength = 100 * tf.abs(D)
#     for i in range(loopLength):
#         sample_selected = Sample(HMDA_df) #Calls the Sample function to randomly get a row (individual) in the dataset
#         sample_similar =
#         similarPairs.append([sample_selected, sample_similar])
#     return similarPairs
#
# # def rankByInfluence(influenceSet, D):
# see [68]


# Notes
# -------------------------------
# Will loop each sample 100 times
# sample_selected = HMDA_df.loc[i]
# for m in range(48842):
# Will lopp each sample again while running the previous loop to find similar sample--or pair
# if euclidianDistance(sample_selected, HMDA_df.loc[m]) < delta:
# HAVE TO ADD SENSITIVE PARAMETER PART
# sample_similar = HMDA_df.loc[m]
# else:
#     print("nothing is similar (which is weird)")
