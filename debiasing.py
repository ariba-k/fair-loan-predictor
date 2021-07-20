from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import time
from random import randint
import pytorch_influence_functions
import main
from pytorch_influence_functions import calc_influence_function
from pytorch_influence_functions import influence_function
from pytorch_influence_functions import __init__

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
from pandas.core.ops import array_ops

# ==========================ABOVE IMPORTS========================================
#:Training dataset D, Sensitive attribute S, Binary
# classification model M trained on D, Input space
# similarity threshold delta
from pytorch_influence_functions.calc_influence_function import calc_s_test, calc_grad_z

HMDA_df = pd.read_csv(r'C:\Users\jasha\PycharmProjects\Debias\adult.csv')

HMDA_df = HMDA_df.dropna()


#
# A function created used to calculate the euclidian distance between two points
def euclidianDistance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)) ** 2)

#Generates a similar individual to the sample individual; same person but with different sensitive params.
def SimilarIndividuals(D, A1, S, Lambda):
    sexes = D[S[0]].unique()
    similar_indivual = A1[:]
    if similar_indivual[9] == sexes[0]:
         similar_indivual[9] = sexes[1]
    elif similar_indivual[9] == sexes[1]:
         similar_indivual[9] = sexes[0]
    return similar_indivual


#Randomly samples the data to create a random individual
def Sample(D):
    #D is the dataset that we are using--in other words, HMDA_df
    numRows = len(D) - 1
    numCols = len(D.columns) - 1
    array = []
    for i in range(numCols):
        numRandom = randint(0, numRows)
        currentRandFeature = D.loc[numRandom].iat[i]
        array.append(currentRandFeature)
    return array






#def sortDataset(D, S, M, delta):
    # This function will generate similar pairs of indivuals for each training sample in a dataset

def generateSimilarPairs(D, S, Lamda):
    similarPairs = []
    loopLength = 100 * len(D)
    for i in range(loopLength):
        sample_selected = Sample(HMDA_df)
        sample_similar = SimilarIndividuals(D, sample_selected, S, Lamda)
        similarPairs.append([sample_selected, sample_similar])
    return similarPairs

#print(generateSimilarPairs(HMDA_df, ["sex"], 0))




''' influenceset is a set of data with the indivuals out of the pairs that were discriminated against; D is dataset'''
def rankByInfluence(influenceset, D):
    split = 0.75
    train_loader = []
    test_loader = []
    main.loadDataset('datasets/adult_5k.data', split, train_loader, test_loader)
    calc_s_test(main.main(), test_loader, train_loader, save=False, gpu=-1, damp=0.01, scale=25, recursion_depth=5000, r=1, start=0)
    print(calc_s_test())
    calc_grad_z(main.main(), train_loader, save_pth=False, gpu=-1, start=0)
    print(calc_grad_z())
    #calc_influence_function(len(D), grad_z_vecs=None,e_s_test=None)

    #I need to get the sum influence of each datapoint on each influenceset prediction and the divide it by total datapoints for each datapoint
    # avg influence for each datapoint
print(rankByInfluence(1, HMDA_df))








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



  # simIndividuals_r2 = []
  # for m in range(27):
  #     for i in range(len(A1)):
  #         if(A1[i] == ethnicities[0] or A1[i] == ethnicities[1] or A1[i] == ethnicities
  #             if (A1[i] == ethnicities[0]):
  #                 startNum1 = 0
  #             if (A1[i] == ethnicities[1]):
  #                 startNum1 = 1
  #             if (A1[i] == ethnicities[2]):
  #                 startNum1 = 2
  #             startNum1 = (startNum1 + 1) % 2
  #             A1[i] = A1[startNum1]
  #         if( A1[i] == races[0] or A1[i] == races[1] or A1[i] == races[2]):
  #             if (A1[i] == races[0]):
  #                 startNum2 = 0
  #             if (A1[i] == races[1]):
  #                 startNum2 = 1
  #             if (A1[i] == races[2]):
  #                 startNum2 = 2
  #             startNum2 = (startNum2 + 1) % 2
  #             A1[i] = A1[startNum2]
  #         if(A1[i] == sexes[0] or A1[i] == sexes[1] or A1[i] == sexes[2]):
  #             if(A1[i] == sexes[0]):
  #                 startNum3 = 0
  #             if (A1[i] == sexes[1]):
  #                 startNum3 = 1
  #             if (A1[i] == sexes[2]):
  #                 startNum3 = 2
  #             startNum3 = (startNum3 + 1) % 2
  #             A1[i] = A1[startNum3]
  #     simIndividuals_r2 = simIndividuals_r2.append(A1)
  # return simIndividuals_r2

       #
       # for k in range(len(S)):
       #     uniqueVal = D[S[k]].unique()
       #     for l in range(len(uniqueVal)):
       #         if(A1[i] == uniqueVal[l]):
       #             currentUnique = uniqueVal[l]
       #             while(uniqueVal[randint(0, len(uniqueVal))] == currentUnique):
       #                 forcedOtherString = uniqueVal[randint(0, len(uniqueVal))]
       #                 A1[i] = forcedOtherString