import numpy as np

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
print(distanceMatrix)
#Calculated distance between two points