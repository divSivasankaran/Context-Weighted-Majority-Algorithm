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
from scipy import spatial

import FisherFace as ff

from shutil import copyfile
import random
import os

K=30
R=90
alpha = 30
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
    print("W_e",W_e.shape)
    print("y",y.shape)
    print("m",m.shape)
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

def generateTemplate_LDA(faces_train,id_train,r=R):
    W_1, b, m  = reduceDimension(faces_train,id_train,r)
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


def getfiles(data,mode,fold,modality,context):
    lst = []
    for line in data:
        line = line.strip()
        ID,f,v,c,k,train,claim = line.split(",")
        if modality == "Voice":
            c = v.split(".")[0].split("_")[-2]
            if mode == "train":
#                if ID not in ["002","006","008","011","017","019","029","035","036","038","042","044","047","048","050","052","055","057"]:
#                    continue
                if c not in context or int(k) == fold or train!="1":
                    continue
                lst.append(line)
            else:
#                if claim not in ["002","006","008","011","017","019","029","035","036","038","042","044","047","048","050","052","055","057"]:
#                    continue
                if int(k)!=fold:
                    continue
                lst.append(line)
        else:
            c = f.split(".")[0].split("_")[-2]
            if mode == "train":
#                if ID not in ["002","006","008","011","017","019","029","035","036","038","042","044","047","048","050","052","055","057","184","186"]:
#                    continue
                if c not in context or int(k) == fold or train!="1":
                    continue
                lst.append(line)
            else:
#                if claim not in ["002","006","008","011","017","019","029","035","036","038","042","044","047","048","050","052","055","057","184","186"]:
#                    continue
                if int(k)!=fold:
                    continue
                lst.append(line)
    return lst


def Voice_Face_Experiment():
    currDir = os.getcwd() + "/../../Datasets/VID"
    #currDir = "C:\Users\e0013178\Google Drive\Study\Course Material\Multimedia Analysis\hw2"
    #Training classifiers
    #context pose
    infile = open("Samples_Paired.csv","r")
    infile.readline() #ignoring header
    data = infile.readlines()  
    data = np.random.permutation(data)
    infile.close()
    for k_fold in range(1,6):
        print(k_fold)
        lst = getfiles(data,"train",k_fold,"Voice",["0","1"])
        print("Training Voice Files: ",len(lst))
        voices_train,id_train,context_train = ff.extractVoiceFeatures(currDir,lst)
        print("Voice",voices_train.shape)
        VC_lda_w,VC_lda_m,VC_lda_feature = generateTemplate_LDA(voices_train,id_train,12)
        print("VLDA_w",VC_lda_w.shape)
        
#        #context pose
        lst = getfiles(data,"train",k_fold,"Face",["2","3"])
        print("Training Pose Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Plda_w,Plda_m,Plda_feature = generateTemplate_LDA(faces_train,id_train)
        print("PLDA_f",Plda_w.shape)
        print("mean",Plda_m.shape)
        
        #context expression
        lst = getfiles(data,"train",k_fold,"Face",["0","1"])
        print("Training Exp Files: ",len(lst))
        faces_train,id_train,context_train = ff.read_faces(currDir,lst)
        Elda_w,Elda_m,Elda_feature = generateTemplate_LDA(faces_train,id_train)
        print("LDA_f",Elda_w.shape)
#        
        lst = getfiles(data,"test",k_fold,"Voice",["0","1"])
        print("testing list",len(lst))
        voices_test,id_test,context_test = ff.extractVoiceFeatures(currDir,lst)
        faces_test,id_test,context_test = ff.read_faces(currDir,lst)
        #Testing classifiers
        filename = "Data_"+ str(k_fold) +".csv"
        fout = open(filename,"w")
        for i in range(0,len(id_test)-1):
            v = voices_test[:,i]
            f = faces_test[:,i]
            l = lst[i]
            Vy_lda = np.dot(VC_lda_w,v-VC_lda_m)
            Ey_lda = np.dot(Elda_w,f-Elda_m)
            Py_lda = np.dot(Plda_w,f-Plda_m)
            claimID = l.split(",")[-1]
            ID =  l.split(",")[0]
            pDist = int(spatial.distance.cosine(Plda_feature[int(claimID)], Py_lda)*50)
            eDist = int(spatial.distance.cosine(Elda_feature[int(claimID)], Ey_lda)*50)
            vDist = int(spatial.distance.cosine(VC_lda_feature[int(claimID)], Vy_lda)*50)
            p_d = 0
            e_d = 0
            v_d = 0
            true_d = int(int(claimID) == int(id_test[i]))
            if pDist <= alpha:
                p_d = 1
            if eDist <= alpha:
                e_d = 1
            if vDist <= alpha:
                v_d = 1
            fout.write((",").join([ID,claimID,str(context_test[i]),str(pDist),str(eDist),str(vDist),str(p_d),str(e_d),str(v_d),str(true_d)]))
            fout.write("\n")
        fout.close()
        newPath = os.getcwd() + "/../../FusionMethods/input/" + filename
        copyfile(filename,newPath)

Voice_Face_Experiment()


