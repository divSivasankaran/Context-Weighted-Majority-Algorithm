# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:46:14 2017

@author: e0013178
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:54:32 2017

@author: e0013178
"""

import numpy as np
from scipy.spatial import distance
import FisherFace as ff
import random
import os
from sklearn.preprocessing import MinMaxScaler

K=30
R=90
alpha = 0.1
np.set_printoptions(threshold=np.nan)

def reduceDimension(faces_train,id_train,r):
    W, LL, m = ff.myPCA(faces_train)
    W_e = W[:,:r]
    y = np.empty([id_train.shape[0],r])
    #calculate PCA feature for each training image
    for i in range(0,id_train.shape[0]):
       f = faces_train[:,i] - m
       y_e = np.dot(np.transpose(W_e),f)
       y[i] = y_e   
    return W_e,y,m

def generateTemplate_PCA(faces_train,id_train):
    W_e,X,m = reduceDimension(faces_train,id_train,K)
    y = dict()
    for i in range(0,X.shape[0]):
        if id_train[i] not in y.keys():
            y[id_train[i]] = []
        y[id_train[i]].append(X[i])
    #create template for each person in the training set
    z = np.empty([len(y.keys()), K])
    for keys in y.keys():
        t = np.mean(np.array(y[keys]),0)
        z[keys] = t
    return W_e,m,z

def generateTemplate_LDA(faces_train,id_train):
    W_1, b, m  = reduceDimension(faces_train,id_train,R)
    X = np.transpose(b)
    W_f,Centers, classLabels = ff.myLDA(X,id_train)
    #calculate LDA feature for each training image
    q = np.transpose(np.dot(np.transpose(W_f),X))
    z = dict()#np.zeros([len(classLabels), q.shape[1]])
    z_count = dict()
    for i in range(0,len(id_train)):
        if id_train[i] not in z.keys():
            z[id_train[i]] = np.zeros(q.shape[1])
            z_count[id_train[i]] = 0
        z[id_train[i]] = z[id_train[i]] + q[i]
        z_count[id_train[i]] = z_count[id_train[i]]+1
    for i in range(0,len(classLabels)):
        z[classLabels[i]] = np.divide(z[classLabels[i]],z_count[classLabels[i]])

    return np.dot(np.transpose(W_f),np.transpose(W_1)),m,z
def getfiles(Dir,mode,fold,context):
    lst = []
    file = open("KFold_Split.csv","r")
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        ID,f,c,k,train,claim = line.split(",")
        
        if mode == "train":
            if ID not in ["002","006","008","011","017","019","029"]:
                continue
            if c not in context or int(k) == fold or train!="1":
                continue
            lst.append(line)
        else:
            if claim not in ["002","006","008","011","017","019","029"]:
                continue
            if int(k)!=fold:
                continue
            lst.append(line)
    file.close()
    return lst
def main():
    currDir = "C:/Users/div_1/Desktop/Scripts/LDA/Dataset"
    #currDir = "C:\Users\e0013178\Google Drive\Study\Course Material\Multimedia Analysis\hw2"
    #Training classifiers
    #context pose
    for k_fold in range(1,3):
        print(k_fold)
        lst = getfiles(currDir,"train",k_fold,["0"])
        print("Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Plda_w,Plda_m,Plda_feature = generateTemplate_LDA(faces_train,id_train)
        print("PLDA_f",Plda_feature[2].shape)
        print("mean",Plda_m.shape)
        
        #context expression
        lst = getfiles(currDir,"train",k_fold,["1"])
        print("Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Elda_w,Elda_m,Elda_feature = generateTemplate_LDA(faces_train,id_train)
        print("LDA_f",Elda_feature[2].shape)
        
        lst = getfiles(currDir,"test",k_fold,["0","1"])
        print("testing list",len(lst))
        faces_test,id_test,context_test = ff.read_faces(currDir,lst)
        #Testing classifiers
        filename = "Data_"+ str(k_fold) +".csv"
        fout = open(filename,"w")
        for i in range(0,len(id_test)-1):
            f = faces_test[:,i]
            l = lst[i]
            Py_lda = np.dot(Plda_w,f-Plda_m)
            Ey_lda = np.dot(Elda_w,f-Elda_m)
            claimID = l.split(",")[-1]
            pDist = distance.euclidean(Plda_feature[int(claimID)], Py_lda)
            eDist = distance.euclidean(Elda_feature[int(claimID)], Ey_lda)
            fout.write((",").join([str(id_test[i]),claimID,str(context_test[i]),str(pDist),str(eDist)]))
            fout.write("\n")
        fout.close()
main()


