import numpy as np
import sys

file = open("../../Desktop/iyer.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t"))-1
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column+1]

true_values = np.array(matrix)[:,0]
matrix = np.array(matrix)
matrix = np.delete(matrix, 0, 1)

#Created a matrix of the data set
#Created an array of true values

def calcuateDistance(p, q):
    sum = 0
    for i in range(len(p)):
        sum += (float(p[i]) - float(q[i]))**2
    return sum ** 0.5

distanceMatrix = [[0 for x in range(rows)] for y in range(rows)]

for i in range(rows):
    for j in range(rows):
        distanceMatrix[i][j] = calcuateDistance(matrix[i], matrix[j])
distanceMatrix = np.asarray(distanceMatrix)

clusterMatrix = np.array(distanceMatrix)
#Calculated distance between two points

clusters = []
for i in range(rows):
    clusters.append([i])

#This method merges lists containing the following numbers
def merge(p1, p2):
    newCluster = []
    newClusters = []
    for cluster in clusters:
        if p1 in cluster and p2 in cluster:
            break
        elif p1 in cluster:
            newCluster += cluster
        elif p2 in cluster:
            newCluster += cluster
        else:
            newClusters.append(cluster)
    newClusters.append(newCluster)
    return newClusters


def findMin(clusterMatrix):
    min = sys.maxsize
    p1 = 0
    p2 = 0
    for row in range(clusterMatrix.shape[0]):
        for column in range(clusterMatrix.shape[1]):
            if clusterMatrix[row][column] == 0.0:
                continue
            if clusterMatrix[row][column] < min:
                flag = False
                for cluster in clusters:                #Here we check if the two points aren't in the same cluster
                    if row in cluster and column in cluster:
                        flag = True
                if flag == True:
                    continue
                else:
                    min = clusterMatrix[row][column]
                    p1 = row
                    p2 = column

    return min, p1, p2

min, p1, p2 = findMin(clusterMatrix)
print(min, ' ', p1, ' ', p2)