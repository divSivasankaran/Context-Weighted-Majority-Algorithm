# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:59:01 2017

@author: e0013178
"""

import numpy as np
import os
import random
from shutil import copyfile

BaseDir = ""
PIE_Exp_Dir = "C:/Users/e0013178/Documents/GitHub/NUS_MThesis/Datasets/Face/multipie_linearCrop/All_exp_Frontal_pose_All_illum"
PIE_Pose_Dir = "C:/Users/e0013178/Documents/GitHub/NUS_MThesis/Datasets/Face/multipie_linearCrop/Neutral_exp_All_poses_All_illum"

#PIE_Exp_Dir = "C:/Users/div_1/OneDrive/Documents/GitHub/NUS_MThesis/Datasets/multipie_linearCrop/All_exp_Frontal_pose_All_illum"
#PIE_Pose_Dir = "C:/Users/div_1/OneDrive/Documents/GitHub/NUS_MThesis/Datasets/multipie_linearCrop/Neutral_exp_All_poses_All_illum"
Sessions = ["session01","session03"]#,"session03","session04"]
OutDir = os.getcwd() + "/../../Datasets/Dataset_57"
totalItems = 57
def mvSamples():
    #file = open("AllExp.csv","w")
    for session in os.listdir(PIE_Exp_Dir):
        if session != Sessions[1]:
            continue
        print("session",session)
        for ID in os.listdir(("/").join([PIE_Exp_Dir,session])):
            if not os.path.exists(OutDir + "/" + ID ):
                os.mkdir(OutDir + "/" + ID )
            for exp in os.listdir(("/").join([PIE_Exp_Dir,session,ID])):
                path = ("/").join([PIE_Exp_Dir,session,ID,exp])
                for img in os.listdir(path):
                    newPath = OutDir + "/" + ID + "/"+ img
                    copyfile(("/").join([path,img]),newPath )
        
   
    for session in Sessions:
        if session == Sessions[1]:
            continue
        print("session",session)
        for ID in os.listdir(("/").join([PIE_Pose_Dir,session])):
            if not os.path.exists(OutDir + "/" + ID ):
                os.mkdir(OutDir + "/" + ID )
            count = 0
            if not os.path.exists(OutDir + "/" + ID ):
                os.mkdir(OutDir + "/" + ID )
            for pose in os.listdir(("/").join([PIE_Pose_Dir,session,ID,"01"])):
                #t =pose.split("_")[-1]
                #if t!= "0":
                if pose not in ["01_0","05_0","19_1"]:    
                    continue
                path = ("/").join([PIE_Pose_Dir,session,ID,"01",pose])
                for img in os.listdir(path):
                    count += 1
                    newPath = OutDir + "/" + ID + "/"+ img
                    copyfile(("/").join([path,img]),newPath )
            print(ID,count)

def createFolds(k):
    outfile = open("FileList.csv","w")
    foldsF = open("KFold_Split.csv","w")
    for ID in os.listdir(OutDir):
        exp_files = []
        pose_files = []
        if ID.split("_")[-1] == "p":
            continue
        files = np.random.permutation(os.listdir(OutDir + "/" + ID))
        for file in files:
            if file.split(".")[-1] != "png":
                continue
            context = 0
            if file.split("_")[0] == "c":
                context = 1
                file = "_".join(file.split("_")[1:])
                exp_files.append(file)
            else:
                pose_files.append(file)
            outfile.write((",").join([ID,file,str(context)]))
            outfile.write("\n")
        items = int(totalItems/k)
        prev = 0
        for i in range(1,k+1):
            data_e = exp_files[prev:(i*items)]
            data_p = pose_files[prev:(i*items)]
            prev = i*items

            for record in data_e:
                foldsF.write((",").join([ID,record,str(1),str(i),str(1),ID]))
                foldsF.write("\n")

            for record in data_p:
                foldsF.write((",").join([ID,record,str(0),str(i),str(1),ID]))
                foldsF.write("\n")
    outfile.close()    
    foldsF.close()
def addImposters(k):
    foldsF = open("KFold_Split.csv","r")
    data = dict()
    for f in foldsF.readlines():
        f = f.strip()
        fold = f.split(",")[-2]
        ID = f.split(",")[0]
        if fold not in data.keys():
            data[fold] = dict()
        if ID not in data[fold].keys():
            data[fold][ID] = []
        data[fold][ID].append(f)
    foldsF.close()
    foldsF = open("KFold_Split.csv","a")
    for f in data.keys():
        fold = data.get(f)
        for ID in fold.keys():
            items = len(fold.get(ID))
            for i in range(1,items):
                new_id = ID
                while(new_id == ID):
                    new_id = random.choice(list(fold.keys())) 
                sample = random.choice(fold[new_id])
                foldsF.write((",").join(sample.split(",")[:4])+",0,"+ID)
                foldsF.write("\n")
    foldsF.close()
#mvSamples()
createFolds(3)
addImposters(3)