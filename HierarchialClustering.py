import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

"""
Read the input file and get the number of rows and columns
"""

file = open("../../Desktop/iyer.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t")) - 1

"""
Create a data matrix consisting of all the gene attributes
Create a separate true_values column containing all the ground truth
"""

k = 2

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

clusters = []
for i in range(rows):
    clusters.append([i])


# This method merges lists containing the following numbers
def merge(p1, p2):
    clusters[p1] = clusters[p1]+clusters[p2]
    clusters.pop(p2)

def updateDistanceMatrix(p1, p2):
    p1 = p1
    p2 = p2
    row1 = distanceMatrix[p1]
    row2 = distanceMatrix[p2]
    row3 = np.minimum(row1, row2)
    col1 = np.transpose(row3)
    distanceMatrix[p1] = row3
    distanceMatrix[:, p1] = col1
    newMatrix = np.delete(distanceMatrix, p2, 0)
    newMatrix = np.delete(newMatrix, p2, 1)
    return newMatrix


def findMin():
    min = sys.maxsize
    p1 = 0
    p2 = 0
    for row in range(distanceMatrix.shape[0]):
        for column in range(row+1, distanceMatrix.shape[1]):
            if distanceMatrix[row][column] < min:
                min = distanceMatrix[row][column]
                p1 = row
                p2 = column
    return min, p1, p2



while(len(clusters) > k):
     min, p1, p2 = findMin()
     distanceMatrix = updateDistanceMatrix(p1, p2)
     merge(p1, p2)
     print(len(clusters))
print(clusters)


def plotOriginalGraph(true_values):
    pca = PCA(n_components=2)
    reduced_Matrix = pca.fit_transform(matrix)
    true_values = np.reshape(true_values, (-1, 1))
    reduced_Matrix = np.hstack((reduced_Matrix, true_values))
    x = [row[0] for row in reduced_Matrix]
    y = [row[1] for row in reduced_Matrix]
    z = [row[2] for row in reduced_Matrix]
    area = np.pi * 15
    plt.scatter(x, y, s=area, c=z,
                cmap='Set1', alpha=1)
    plt.title('Scatter plot with reduced dimensionality')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.legend()
    plt.show()

plotOriginalGraph(true_values)


groundTruth = [[0 for x in range(rows)] for y in range(rows)]
groundTruth = np.array(groundTruth)
for row in range(groundTruth.shape[0]):
    for column in range(groundTruth.shape[1]):
        if true_values[row] == true_values[column]:
            groundTruth[row][column] = 1
        else:
            groundTruth[row][column] = 0

clusterTruth = [[0 for x in range(rows)] for y in range(rows)]
clusterTruth = np.array(clusterTruth)
for row in range(rows):
    for column in range(rows):
        for cluster in clusters:
            if row+1 in cluster and column+1 in cluster:
                clusterTruth[row][column] = 1

m11 = 0
m00 = 0
m01 = 0
m10 = 0
for row in range(groundTruth.shape[0]):
    for column in range(groundTruth.shape[1]):
        if groundTruth[row][column] == 1 and clusterTruth[row][column] == 1:
            m11+=1
        elif groundTruth[row][column] == 0 and clusterTruth[row][column] == 0:
            m00+=1
        elif groundTruth[row][column] == 0 and clusterTruth[row][column] == 1:
            m01+=1
        elif groundTruth[row][column] == 1 and clusterTruth[row][column] == 0:
            m10+=1

rIndex = (abs(m11) + abs(m00))/(abs(m00) + abs(m11) + abs(m01) + abs(m10))
jCoefficient = abs(m11)/(abs(m11) + abs(m10) + abs(m01))
print('Rand Index: ',rIndex)
print('Jaccard Coefficient: ', jCoefficient)

for i in range(len(clusters)):
    for element in clusters[i]:
        true_values[element] = i+1

plotOriginalGraph(true_values)