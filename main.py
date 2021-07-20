# this is from internet
# import the dependencies
import csv
import random
import math
import operator
import sys
import time


# load datasets into matrix form where x is row, and y is column
# params:
# - filename: file name that contains the datasets
# - split: percentage of training data, it's between 0.75, 0.8, or 0.9
# output:
# - trainingSet: training data
# - testSet: test data
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        maxTrainData = int(split * len(dataset))
        for x in range(0, len(dataset)):
            for y in range(14):
                dataset[x][y] = float(dataset[x][y])
            if x < maxTrainData:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                return trainingSet,testSet


# this function is for calculating euclidan distance
# between 2 points (every of train data and test data).
# this is a basic of pythagoras.
# params:
# - instance1: test instance that cotains all test data
# - instance2: 1 row of training data
# - length: length of features - 1 (for loop)
# output:
# - euclidean distance
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# get testInstance nearest neighbors in trainingSet
# params:
# - trainingSet: set of training data
# - testInstance: instance of test data that we'll going to find his nearest neighbors
# output:
# - k-neighbors data
def getNeighbors(trainingSet, testInstance, k):
    distances = []

    # get the lengh - 1, because the last one is not data, but definition
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        # calculate euclidean distance of test instance and training instance,
        # they have 'length' of columns
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # collect training instance and it's distance to an array
        distances.append((trainingSet[x], dist))

    # sort the array of distance by it's distance's value
    distances.sort(key=operator.itemgetter(1))

    # get k-nearest neigbors
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# count the most nearest neighbors
# params:
# - neighbors: neigbors data
# output:
# - the definition, based on the most nearest neigbors
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    # sort classVotes by it's value descending, to get the maximum one
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    # return the first one
    return sortedVotes[0][0]


# calculate the percentage of program's accuracy
# comparing the result of predictions to testSet
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# main program
def main():
    # get the start time
    startTime = time.time()

    trainingSet = []
    testSet = []

    # default value of split is 0.75, it will be replaced if the split argument is passed
    split = 0.75
    if (len(sys.argv) > 1):
        if sys.argv[1] == "0.75" or sys.argv[1] == "0.8" or sys.argv[1] == "0.9":
            split = float(sys.argv[1])
        else:
            print('WARNING:\nRatio\'s accepted value is 0.75, 0.8, or 0.9\nUsing default instead')

    # load data set
    loadDataset('datasets/adult_5k.data', split, trainingSet, testSet)

    # print info
    print('Program Starts\n')
    print('Train ratio: ' + repr(split))
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    print('Calculating ...')

    predictions = []

    # num of neighbors that will be counted
    k = 3

    for x in range(len(testSet)):
        # get neighbors as much as k
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        # determine the prediction
        result = getResponse(neighbors)
        # add the prediction into array
        predictions.append(result)

    # calculate the accuracy
    accuracy = getAccuracy(testSet, predictions)

    # print info
    print('Accuracy: ' + repr(round(accuracy, 2)) + '%')
    print('Execution time: ' + str(round(time.time() - startTime, 2)) + ' seconds')
    print('\nProgram Ends')


main()