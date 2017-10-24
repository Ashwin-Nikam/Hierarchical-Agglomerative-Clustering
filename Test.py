import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys

"""
Read the input file and get the number of rows and columns
"""

file = open("../../Desktop/cho.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t")) - 1

"""
Create a data matrix consisting of all the gene attributes
Create a separate true_values column containing all the ground truth
"""

matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column + 1]
true_values = np.array(matrix)[:, 0]
matrix = np.array(matrix)
matrix = np.delete(matrix, 0, 1)

distanceMatrix = [[0.0 for x in range(rows)] for y in range(rows)]
distanceMatrix = np.asarray(distanceMatrix)

distanceMatrix = euclidean_distances(matrix, matrix)

def updateDistanceMatrix(cluster1, cluster2):
    newRow = np.empty([1, rows])
    min = sys.maxsize;
    for i in range(distanceMatrix.shape[0]):
        for item in cluster1:
            distance = euclidean_distances(matrix[item], matrix[i])
            if distance < min:
                min = distance
        newRow.append(distance)
    newColumn = np.transpose(newRow)
    distanceMatrix[cluster1[0]] = newRow
    distanceMatrix[:,cluster1[0]] = newColumn
    np.delete(distanceMatrix, cluster2[0], cluster2[0])


clusters = []
for i in range(rows):
    clusters.append([i]);


# This method merges lists containing the following numbers
def merge(p1, p2):
    newCluster = []
    for cluster in clusters:
        if p1 in cluster and p2 in cluster:
            break
        elif p1 in cluster:
            newCluster += cluster
        elif p2 in cluster:
            newCluster += cluster
    list.sort(newCluster)
    clusters.pop(p2)
    clusters[newCluster[0]] = newCluster

print(matrix[0])
print(matrix[1])
print(distanceMatrix)
updateDistanceMatrix([0], [1])
print(distanceMatrix)

# print(clusters)
# merge(0, 1)
# print(clusters)
# merge(4,5)
# print(clusters)