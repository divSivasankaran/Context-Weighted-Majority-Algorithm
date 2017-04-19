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
from scipy import spatial

K=30
R=90
alpha = 0.6
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
def getfiles(data,mode,fold,context):
    lst = []
    for line in data:
        line = line.strip()
        ID,f,c,k,train,claim = line.split(",")
        
        if mode == "train":
#            if ID not in ["002","006","008","011","017","019","029"]:
#                continue
            if c not in context or int(k) == fold or train!="1":
                continue
            lst.append(line)
        else:
#            if claim not in ["002","006","008","011","017","019","029"]:
#                continue
            if int(k)!=fold:
                continue
            lst.append(line)
    return lst
def main():
    currDir = "C:/Users/div_1/Desktop/Scripts/LDA/Dataset_57"
    #currDir = "C:\Users\e0013178\Google Drive\Study\Course Material\Multimedia Analysis\hw2"
    #Training classifiers
    #context pose
    c = dict()
    current_context_count = 1
    infile = open("KFold_Split.csv","r")
    data = infile.readlines()  
    data = np.random.permutation(data)
    infile.close()
    for k_fold in range(1,4):
        print(k_fold)
        lst = getfiles(data,"train",k_fold,["0"])
        print("Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Plda_w,Plda_m,Plda_feature = generateTemplate_LDA(faces_train,id_train)
        print("PLDA_f",Plda_feature[2].shape)
        print("mean",Plda_m.shape)
        
        #context expression
        lst = getfiles(data,"train",k_fold,["1"])
        print("Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Elda_w,Elda_m,Elda_feature = generateTemplate_LDA(faces_train,id_train)
        print("LDA_f",Elda_feature[2].shape)
         
        #train on all data - trying to be best at everything!
        lst = getfiles(data,"train",k_fold,["0","1"])
        print("Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Glda_w,Glda_m,Glda_feature = generateTemplate_LDA(faces_train,id_train)
        print("LDA_f",Elda_feature[2].shape)
        
        
        lst = getfiles(data,"test",k_fold,["0","1"])
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
            Gy_lda = np.dot(Glda_w,f-Glda_m)
            claimID = l.split(",")[-1]
            c_id,c_session,c_exp,c_pose,c_illm = l.split(",")[1].split("_")
            if str(c_exp)+str(c_pose) not in c.keys():
                c[ str(c_exp)+str(c_pose)] = current_context_count
                current_context_count += 1
            pDist = spatial.distance.cosine(Plda_feature[int(claimID)], Py_lda)
            eDist = spatial.distance.cosine(Elda_feature[int(claimID)], Ey_lda)
            gDist = spatial.distance.cosine(Glda_feature[int(claimID)], Gy_lda)
            p_d = 0
            e_d = 0
            g_d = 0
            true_d = int(claimID == c_id)
            if pDist <= alpha:
                p_d = 1
            if eDist <= alpha:
                e_d = 1
            if gDist <= alpha:
                g_d = 1
            fout.write((",").join([str(id_test[i]),str(claimID),str(context_test[i]),str(pDist),str(eDist),str(gDist),str(c[ str(c_exp)+str(c_pose)]),str(p_d),str(e_d),str(g_d),str(true_d)]))
            fout.write("\n")
        fout.close()
main()


