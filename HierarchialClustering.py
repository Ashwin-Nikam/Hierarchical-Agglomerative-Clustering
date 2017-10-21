import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

"""
Create a ground truth matrix
"""

groundTruth = [[0 for x in range(rows)] for y in range(rows)]
groundTruth = np.array(groundTruth)
for row in range(groundTruth.shape[0]):
    for column in range(groundTruth.shape[1]):
        if true_values[row] == true_values[column]:
            groundTruth[row][column] = 1
        else:
            groundTruth[row][column] = 0

distanceMatrix = [[0 for x in range(rows)] for y in range(rows)]
distanceMatrix = np.asarray(distanceMatrix)


def calculateDistance(p, q):
    sum = 0
    for i in range(len(p)):
        sum += (float(p[i]) - float(q[i])) ** 2
    return sum ** 0.5


for i in range(distanceMatrix.shape[0]):
    for j in range(distanceMatrix.shape[1]):
        distanceMatrix[i][j] = calculateDistance(matrix[i], matrix[j])

clusterMatrix = np.array(distanceMatrix)
clusters = []
for i in range(rows):
    clusters.append([i])


# This method merges lists containing the following numbers
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


def findMin():
    min = sys.maxsize
    p1 = 0
    p2 = 0
    for row in range(clusterMatrix.shape[0]):
        for column in range(clusterMatrix.shape[1]):
            if clusterMatrix[row][column] == 0.0:
                continue
            if clusterMatrix[row][column] < min:
                flag = False
                for cluster in clusters:  # Here we check if the two points aren't in the same cluster
                    if row in cluster and column in cluster:
                        flag = True
                if flag == True:
                    continue
                else:
                    min = clusterMatrix[row][column]
                    p1 = row
                    p2 = column

    return min, p1, p2


# while(len(clusters) > 1):
#     min, p1, p2 = findMin()
#     clusters = merge(p1, p2)
#     print(len(clusters))
# print(clusters)

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
