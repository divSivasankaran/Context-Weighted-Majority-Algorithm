# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:15:58 2017

@author: e0013178
"""

import numpy as np
import os
import random
from shutil import copyfile

Base_Dir = "C:/Users/e0013178/Documents/GitHub"
Face_Dir = Base_Dir + "/NUS_MThesis/Datasets/Face/multipie_linearCrop/All_exp_Frontal_pose_All_illum/session03"
Face_PoseDir = Base_Dir + "/NUS_MThesis/Datasets/Face/multipie_linearCrop/Neutral_exp_All_poses_All_illum/session01"
Gait_Dir = Base_Dir + "/NUS_MThesis/Datasets/Gait/zju-gaitacc/session_"
Voice_Dir = Base_Dir + "/CWMA/Datasets/ID_Clean_Noise_Splitted_Filtered"
Dataset_Dir = os.getcwd() + "/../../Datasets/VID"
globalContext = []
## Use this function to list out & create virtual ids from random permutations
def generateVirtualID():
    
    #gait = np.arange(gait_size)
    #voice = np.arange(voice_size)
    #face = np.arange(face_size)
   # gait = []
    voice = []
    face = []
#    for f in os.listdir(Gait_Dir):
#        gait.append(f)
#    
    for f in os.listdir(Voice_Dir):
        if f not in voice and f.split(".")[-1]!="mat":
            voice.append(f)
    
    for f in os.listdir(Face_Dir):
        face.append(f)
        
    file = open("OriginalIDList.csv","w")
    size = min(len(face),len(voice))#len(gait))
#    file.write("Gait:,")
#    file.write((",").join(str(e) for e in gait))
#    file.write("\n")
#    
    file.write("Face:,")
    file.write((",").join(str(e) for e in face))
    file.write("\n")
    file.write("Voice:,")
    file.write((",").join(str(e) for e in voice))
    file.write("\n")
    file.close()
    file = open("VirtualIDList.csv","w")
    
#    gait = np.random.permutation(gait)
    voice = np.random.permutation(voice)
    face = np.random.permutation(face)
    file.write("VirtualID,FaceID,VoiceID\n")#,GaitID\n")
    for i in range(0,size):
        file.write((",").join(['{num:03d}'.format(num=i),str(face[i]),str(voice[i])]))#,str(gait[i])]))
        file.write("\n")
    file.close()


## Use this function to move sample data to the virtualID folders in the new structure
def mvToVirtualIDStruc():
    file = open("VirtualIDList.csv","r")
    voice_file = open("SITWFileNames.csv","w")
    face_file = open("PIEFileNames.csv","w")
#    gait_file = open("ZJUFileNames.csv","w")
    #ignoring header
    file.readline()
    lines = file.readlines()
    face_contexts = ["03","02","05_0","01_0"]
    for line in lines:
        vID,faceID,voiceID = line.strip().split(",")
        # read each file from the original locations
        person_location = ("/").join([Dataset_Dir,vID])
        if not os.path.exists(person_location):
            os.mkdir(person_location)
            os.mkdir(("/").join([person_location,"voice"]))
            os.mkdir(("/").join([person_location,"face"]))
#            os.mkdir(("/").join([person_location,"gait"]))
        #Moving PIE/FaceFiles
        f_loc = ("/").join([Face_Dir,faceID])
        count = 0
        #Expression change
        for f in os.listdir(f_loc):
            if f in face_contexts[:2]:
                for c in os.listdir(("/").join([f_loc,f])):
                    #verifying they are image files
                    if not c.split(".")[-1] =='png':
                        continue
                    # write oldfilename,newfilename 
                    newName = ("_").join([vID,"f",str(face_contexts.index(f)),str(count)]) + ".png"
                    count += 1
                    face_file.write((",").join([c,newName]))
                    face_file.write("\n")

                    # move and rename file
                    copyfile(("/").join([f_loc,f,c]), ("/").join([person_location,"face",newName]))
        f_loc = ("/").join([Face_PoseDir,faceID,"01"])
        #Pose change
        
        for f in os.listdir(f_loc):
            if f in face_contexts[2:]:
                for c in os.listdir(("/").join([f_loc,f])):
                    #verifying they are image files
                    if not c.split(".")[-1] =='png':
                        continue
                    # write oldfilename,newfilename 
                    newName = ("_").join([vID,"f",str(face_contexts.index(f)),str(count)]) + ".png"
                    count += 1
                    face_file.write((",").join([c,newName]))
                    face_file.write("\n")
                    # move and rename file
                    copyfile(("/").join([f_loc,f,c]), ("/").join([person_location,"face",newName]))
        #Moving SITW Files
        count = 0
        for f in os.listdir(Voice_Dir):
            if f == voiceID:
                i = 0
                limit = 0 
                for c in os.listdir(("/").join([Voice_Dir,f])):
                    for p in os.listdir(("/").join([Voice_Dir,f,c])):
                        if i>0 and count>limit*2-1: #ensure that the number of clean and noisy samples per peson is the same
                            continue
                        # write oldfilename,newfilename 
                        newName = ("_").join([vID,"v",str(i),str(count)]) + ".wav"
                        count += 1
                        voice_file.write((",").join([p,newName]))
                        voice_file.write("\n")
                        
                        copyfile(("/").join([Voice_Dir,f,c,p]), ("/").join([person_location,"voice",newName]))
                    i += 1
                    limit = count
##        #Moving ZJU-Gait Files
#        sessions = ["1","2"]
#        for s in sessions:
#            for f in os.listdir(Gait_Dir+s+"/"+gaitID):
#                if f=="total.txt":
#                    continue
#                for r in os.listdir(("/").join([Gait_Dir+s,gaitID,f])):
#                    if r in ["useful.txt","cycles.txt"]:
#                        continue
#                    path = ("/").join([Gait_Dir+s,gaitID,f,r])
#                    newName = newName = ("_").join([vID,"g",r,str(count)])
#                    count += 1
#                    gait_file.write((",").join([path,newName]))
#                    gait_file.write("\n")
#                    copyfile(path, ("/").join([person_location,"gait",newName]))
                    
                    
        
    voice_file.close()
    face_file.close()       
    
#helper function
def getContext(f,v):#.g)
    f = f.split(".")[0]
    f = f.split("_")[-2]
    v = v.split(".")[0]
    v = v.split("_")[-2]
#    g = g.split(".")[0]
#    g = g.split("_")[-2]
    if f+v not in globalContext:
        globalContext.append(f+v)
    return str(globalContext.index(f+v))
# Use this function to pair the sample data
def pairSamples(k):
    file = open("Samples_Paired.csv","w")
    file.write("VirtualID,FaceLocation,VoiceLocation,Context,Fold,Genuine,ClaimID\n")
    for vID in os.listdir(Dataset_Dir):
         #for each person, pair the number of face, gait & voice samples we have so far!
         face = []
#         gait = []
         voice = []
         for f in os.listdir("/".join([Dataset_Dir,vID,"face"])):
             face.append(f)
         for f in os.listdir("/".join([Dataset_Dir,vID,"voice"])):
             voice.append(f)
#         for f in os.listdir("/".join([Dataset_Dir,vID,"gait"])):
#             gait.append(f)
         Samples = min(len(face),len(voice))#,len(gait))
#         gait = np.zeros(globalSamples)

#         gait = np.random.permutation(gait)
         voice = np.random.permutation(voice)
         face = np.random.permutation(face)
         items = int(Samples/k)
         currK = 1
         counter = 0
         for i in range(0,Samples):
             if counter == items:
                 counter = 0
                 currK += 1
             if currK > k:
                 break
             file.write((",").join([vID,str(face[i]),str(voice[i]),getContext(face[i],voice[i]),str(currK),str(1),vID])) #,gait[i])
             file.write("\n")
             counter += 1
    file.close()

def addImposters(k):
    foldsF = open("Samples_Paired.csv","r")
    data = dict()
    foldsF.readline() #Ignoring header
    for f in foldsF.readlines():
        f = f.strip()
        fold = f.split(",")[-3]
        ID = f.split(",")[0]
        if fold not in data.keys():
            data[fold] = dict()
        if ID not in data[fold].keys():
            data[fold][ID] = []
        data[fold][ID].append(f)
    foldsF.close()
    foldsF = open("Samples_Paired.csv","a")
    for f in data.keys():
        fold = data.get(f)
        for ID in fold.keys():
            items = len(fold.get(ID))
            for i in range(1,items+1):
                new_id = ID
                while(new_id == ID):
                    new_id = random.choice(list(fold.keys())) 
                sample = random.choice(fold[new_id])
                foldsF.write((",").join(sample.split(",")[:5])+",0,"+ID)
                foldsF.write("\n")
    foldsF.close()

### Step 1
generateVirtualID()

### Step 2
mvToVirtualIDStruc()
k=5
### Step 3
pairSamples(k)

### Step 4
addImposters(k)


