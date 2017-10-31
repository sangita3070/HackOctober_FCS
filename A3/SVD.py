import os
import numpy as np
from copy import deepcopy


d = 0
totalMovies = 5
totalUsers = 5
train_data_matrix = np.zeros((totalUsers,totalMovies))
train_data_matrix_dummy = np.zeros((totalUsers,totalMovies))
print os.getcwd()
os.chdir("C:\Users\AJAY\PycharmProjects\CF_Test_UserUser")
print os.getcwd()

def readTrainData():

    global train_data_matrix
    global train_data_matrix_dummy
    with open("u2.txt") as myFile:
        for line in myFile:
            line = line.split("\t")
            user=int(line[0])-1
            movie=int(line[1])
            movie = movie-1
            rating=int(line[2])
            train_data_matrix[user, movie] = rating
    train_data_matrix_dummy = deepcopy(train_data_matrix)
    train_mean = np.mean(train_data_matrix_dummy)
    train_data_matrix_dummy[train_data_matrix_dummy==0]=train_mean

def svd_method():

    count = 3
    while count > 0:
        global d
        global train_data_matrix_dummy
        global train_data_matrix
        U, s, V = np.linalg.svd(train_data_matrix_dummy, full_matrices=True)
        U[U < 0] = 0
        V[V < 0] = 0
        d = len(s)
        S = np.zeros((len(U), d))
        S = np.diag(s)
        RL1 = np.dot(U[:, 1:d], S[1:d, 1:d])
        RL2 = np.dot(RL1, np.transpose(V[:, 1:d]))
        printing_new_matrix(RL2)
        print 'RL* final matrix'
        print RL2
        train_data_matrix_dummy = []
        train_data_matrix_dummy = deepcopy(RL2)
        count = count - 1


def printing_new_matrix(RL2):

    global train_data_matrix
    rows = len(train_data_matrix)
    columns = len(train_data_matrix[0])
    for i in range(rows):
        for j in range(columns):
            if train_data_matrix[i][j] != 0:
                RL2[i][j] = train_data_matrix[i][j]


readTrainData()
svd_method()
