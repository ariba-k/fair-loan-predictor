# Judul: Implementasi k-Nearest Neighbor Terhadap Adult Dataset Menggunakan Bahasa Pemrograman Python
# Deskripsi: Untuk memenuhi tugas besar mata kuliah Artificial Intelligence
# Anggota Kelompok:
# - 0617124001 Asep Maulana Ismail
# - 0617124007 Annas Shibghahtullah
# - 0617124013 Roy Artama Saragi
import csv
import math
import operator 
import time
import sys

def generateData(filename, split):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        trainLength = int(len(dataset)*split)
        for x in range(0, len(dataset)):
            for y in range(len(dataset[x])):
                dataset[x][y] = int(dataset[x][y])
        return dataset[:trainLength], dataset[trainLength:]

def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)-1):
        distance += pow(((instance1[x]) - (instance2[x])), 2)
    return math.sqrt(distance)

def getKNN(trainSet, testObject, k):
    distances = []
    for i in range(len(trainSet)):
        dist = euclideanDistance(trainSet[i], testObject)
        distances.append((trainSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    return distances[:k]

def getTopDefinition(knn):
    classVotes = {}
    for i in range(len(knn)):
        response = knn[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][1]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if (testSet[x][-1]) == predictions[x]:
            correct += 1

    return (correct/float(len(testSet))) * 100.0


def main():
    startTime = time.time()
    split = 0.75
    if (len(sys.argv) > 1):
        if sys.argv[1] == "0.75" or sys.argv[1] == "0.8" or sys.argv[1] == "0.9":
            split = float(sys.argv[1])
        else:
            print('WARNING:\nRatio\'s accepted value is 0.75, 0.8, or 0.9\nUsing default instead')

    k = 3
    predictions = []
    trainSet, testSet = generateData("datasets/adult_full.data", float(sys.argv[1]))
    # print info
    print('Program Starts\n')
    print('Train ratio: ' + repr(split))
    print('Train set: ' + repr(len(trainSet)))
    print('Test set: ' + repr(len(testSet)))
    print('Calculating ...')
    for i in range(len(testSet)):
        knn = getKNN(trainSet, testSet[i], k)
        prediction = getTopDefinition(knn)
        predictions.append(prediction)
        print("Progress: " + repr(round((i/len(testSet))*100, 2)) + "%")
    
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(round(accuracy,2)) + '%')
    print('Execution time: ' + repr(round(time.time() - startTime, 2)) + ' seconds')
    print('\nProgram Ends')

main()