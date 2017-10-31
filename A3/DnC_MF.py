import numpy as np
from copy import deepcopy
import os
from numpy import linalg as LA

totalMovies = 1682
totalUsers = 943
Y = np.zeros((totalUsers, totalMovies))
R = np.zeros((totalUsers, totalMovies))
count = 5
k = 2

os.chdir("C:\Users\AJAY\PycharmProjects\CF_Test_UserUser")

def readTrainData(Y):
    with open("u1.base.txt") as myFile:
        for line in myFile:
            line = line.split("\t")
            user = int(line[0])-1
            movie = int(line[1])
            movie = movie-1
            rating = int(line[2])
            Y[user, movie] = rating
    print 'Y'
    print Y
    XO = deepcopy(Y)
    return XO

def initialize(XO, R, Y):
    for i in range(len(XO)):
        for j in range(len(XO[i])):
            if XO[i][j] == 0:
                row = np.mean(Y[i, :])
                #print row
                column = np.mean(Y[:, j])
                #print column
                mean_ = (row + column)/2.0
                XO[i][j] = mean_
            else:
                R[i][j] = 1

def divideMatrices(matrix):
    mat1 = matrix[:, 0:420]
    mat2 = matrix[:, 420:840]
    mat3 = matrix[:, 840:1260]
    mat4 = matrix[:, 1260:1682]
    return mat1, mat2, mat3, mat4

def latent_method(mat1, R, Y):
    global count
    count = 20
    while count > 0:
        print count
        count = count - 1
        B = mat1 + (Y - np.multiply(R, mat1))
        UK, VK = matrix_factorisation(B)
        mat1 = np.dot(UK, VK)
    return mat1

def ista(B, lamb, U):
    global k
    VO = np.zeros((B.shape[1], k))
    VK = np.zeros((B.shape[1], k))
    w, v = LA.eig(np.dot(U, np.transpose(U)))
    alpha = np.amax(w)
    for loop in range(10):
        T = VO + np.transpose((1/alpha)*(np.dot(np.transpose(U), (B - np.dot(U, np.transpose(VO))))))
        for i in range(len(T)):
            for j in range(len(T[0])):
                #VK[i][j] = np.real(np.sign(T[i][j])*np.maximum(0, abs(T[i][j])-(lamb/(2*alpha))))
                VK[i][j] = lamb * np.real(np.sign(T[i][j]) * np.maximum(0, np.absolute(T[i][j]) - (lamb / (2 * alpha))))
        VO = VK
    return VK

def matrix_factorisation(B):
    count1 = 10
    global k
    global totalUsers
    lamb_reg = [0.1] * k
    lamb = 0.1
    U = np.random.randint(low=1, high=5, size=(len(B), k))
    while count1 > 0:
        count1 = count1 - 1
        U_trans = np.transpose(U)
        VK = np.linalg.lstsq(np.dot(U_trans, U), np.dot(np.transpose(U), B))[0] + np.transpose(ista(B, lamb, U))
        VK[VK < 0] = 0
        VK_trans = np.transpose(VK)
        R_trans = np.transpose(B)
        UK_temp = np.linalg.lstsq(np.dot(VK, VK_trans) + np.diag(lamb_reg), np.dot(VK, R_trans))[0]
        UK = np.transpose(UK_temp)
        UK[UK < 0] = 0
        U = UK
        if np.array_equal(np.dot(UK, VK), B) == True:
            break
    return UK, VK
    #return np.dot(UK, VK)

def testing (ratings):

    records = 0
    total_mae = 0.0
    print 'testing method'
    os.chdir("C:\Users\AJAY\PycharmProjects\CF_Test_UserUser")
    with open("u1.test.txt") as myFile:
        for line in myFile:
            records = records + 1
            line = line.split("\t")
            user = int(line[0]) - 1
            movie = int(line[1])
            movie = movie - 1
            actual_rating = int(line[2])
            predicted_rating = ratings[user][movie]
            #print actual_rating, predicted_rating
            total_mae = total_mae + abs(predicted_rating - actual_rating)
    #print 'mae'
    mae = total_mae/records
    print mae
    print 'nmae'
    print mae/4.0


XO = readTrainData(Y)
print 'done reading training data'
initialize(XO, R, Y)
print 'done initializing'
X1, X2, X3, X4 = divideMatrices(XO)
Y1, Y2, Y3, Y4 = divideMatrices(Y)
R1, R2, R3, R4 = divideMatrices(R)
print 'divide of matrices is completed'
latent1 = latent_method(X1, R1, Y1)
print 'latent 1 completed'
latent2 = latent_method(X2, R2, Y2)
print 'latent 2 completed'
latent3 = latent_method(X3, R3, Y3)
print 'latent 3 completed'
latent4 = latent_method(X4, R4, Y4)
print 'latent 4 completed'
print 'latent factors calculations are completed'
L12 = np.dot(np.dot(latent1, np.linalg.pinv(latent1)), latent2)
L13 = np.dot(np.dot(latent1, np.linalg.pinv(latent1)), latent3)
L14 = np.dot(np.dot(latent1, np.linalg.pinv(latent1)), latent4)
print 'projections calculations completed'
ratings = np.concatenate((latent1, L12, L13, L14), axis=1)
print 'ratings matrix completed'
print np.shape(ratings)
testing(ratings)
