# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 14:44:29 2017

@author: div_1
"""
import numpy as np
import os

outdir = os.getcwd() + "\\..\\output\\stat.csv"
indir = os.getcwd() + "\\..\\input"      

class WMA(object):
    def __init__(self, experts = 0):
        self.mProbability = np.empty(experts)
        self.mWeights = np.empty(experts)
        
        if experts!=0:
            self.mProbability.fill(float(1.0)/float(experts))
        self.mWeights.fill(1)
        self.mExpertLoss = np.zeros(experts)
        self.mHits = np.zeros(experts)
                
        self.mBestActionLoss = 0.0
        self.mLoss = 0.0
        self.setExperts(experts)
        self.mRounds = 0
        self.EPSILON = 0.5
        
    def setExperts(self,experts):
        self.mExperts = experts
        
    def getHitVector(self):
        return self.mHits
    
    def getLoss(self):
        return self.mLoss
    
    def getWeights(self):
        return self.mWeights
    
    def getBestExpertLoss(self):
        return min(self.mExpertLoss)
    
    def getBestActionLoss(self):
        return self.mBestActionLoss
    
    def getExpertDecision(self,expert_decisions, actual_decision):
        majority = 0
        bestAction = False
        for i in range(0,self.mExperts):
            if expert_decisions[i] == 1 :
                majority += self.mProbability[i]
            else:
                majority -= self.mProbability[i]
            if(expert_decisions[i]==actual_decision):
                bestAction = True
        if bestAction == False:
            self.mBestActionLoss += 1
        
        if majority>=0:
            return 1.0
        else:
            return 0.0
        
    def updateWeights(self, expert_decisions,actual_decision):
        decision = self.getExpertDecision(expert_decisions,actual_decision)
        if decision != actual_decision:
            self.mLoss += 1
        total_weight = 0
        for i in range(0,self.mExperts):
            if expert_decisions[i] != actual_decision:
                self.mWeights[i] = self.mWeights[i] * np.exp(-self.EPSILON)
                self.mExpertLoss += 1
            else: 
                self.mHits[i] += 1
            total_weight += self.mWeights[i]
        
        for i in range(0, self.mExperts):
            self.mProbability[i] = self.mWeights[i]/total_weight
        return decision
    
    def train(self, expert_decisions,actual_decisions):
        outfile = open(outdir,"w")
        outfile.write("WMA,avgLoss,currentLoss,BestActionLoss,BestExpertLoss\n")
        self.__init__(expert_decisions.shape[1])
        self.mRounds = actual_decisions.shape[0] 
        for i in range(0,self.mRounds):
            #if i == 5:
              #  break
            self.updateWeights(expert_decisions[i],actual_decisions[i])    
            outfile.write((",").join([str(i),str(self.mLoss/(i+1)),str(self.mLoss),str(self.getBestActionLoss()),str(self.getBestExpertLoss()),"\n"]))
        outfile.close()
    
    def write(self,ofile):
        ofile.write("Total Loss: " + str(self.mLoss) + "\n")
        ofile.write("Average Loss: " + str(self.mLoss/(self.mRounds+1))+"\n")
        ofile.write("Expert,Weight,Hits,Misses\n")
        for i in range(0,self.mExperts):
            ofile.write((",").join([str(i),str(self.mWeights[i]),str(self.mHits[i]/self.mRounds),str(self.mExpertLoss[i]),"\n"]))

class CWMA(object):
    def __init__(self,experts=0,contexts=0):
        self.setExperts(experts)
        self.setContexts(contexts)
        self.mWMA = []
        for i in range(0,contexts):
            self.mWMA.append(WMA(experts))
        self.mLoss = 0
        self.mBestLoss = 0
        self.mRounds = 0
        
    def setExperts(self,experts):
        self.mExperts = experts
    
    def setContexts(self,contexts):
        self.mContexts = contexts
    
    def getHitMatrix(self):
        hitmatrix = np.empty(self.mContexts)
        for c in range(0, self.mContexts):
            hitV = self.mWMA[c].getHitVector()
            for i in range(0, self.mExperts):
                if(self.mWMA[c].mRounds!=0):
                    hitV[i] = hitV[i]/self.mWMA[c].mRounds
            hitmatrix[c] = hitV
    
    def getLoss(self):
        return self.mLoss
    
    def getBestLoss(self):
        mBestLoss = 0
        for i in range(0,self.mContexts):
            mBestLoss += self.mWMA[i].getBestExpertLoss()
        
        return mBestLoss
    
    def write(self,ofile):
        ofile.write("Total Loss of CWMA: " + str(self.mLoss)+ "\n")
        ofile.write("Average Loss: " + str(self.mLoss/self.mRounds)+"\n")
        for i in range(0, self.mContexts):
            ofile.write("Characteristics of experts in context " + str(i) + "\n")
            self.mWMA[i].write(ofile)
    
    def train(self, expert_decisions,actual_decisions,contexts):
        self.__init__(expert_decisions.shape[1], np.unique(contexts).shape[0])        
        self.mRounds = contexts.shape[0]
        outfile = open(outdir,"a")
        outfile.write("CWMA,avgLoss,currentLoss,BestContextLoss\n")
        
        for i in range(0,self.mRounds):
            if expert_decisions.shape[1] != self.mExperts:
                return
            self.updateWeights(expert_decisions[i],actual_decisions[i],contexts[i])
            outfile.write((",").join([str(i),str(self.mLoss/(i+1)),str(self.mLoss),str(self.getBestLoss()),"\n"]))
        
        outfile.close()
    
    def updateWeights(self,expert_decisions,actual_decision,context):
        self.mWMA[context].mRounds += 1
        decision = self.mWMA[context].updateWeights(expert_decisions,actual_decision)
        if(decision!=actual_decision):
            self.mLoss += 1
        return decision
        
def main():
    infile = open(indir+"\\Data_2.csv", "r")
    lines = infile.readlines()
    actual_decisions = np.empty(0).astype(int)
    expert_decisions = []
    contexts = np.empty(0).astype(int)
    contextMap = dict()
    contextCount = 0
    for line in lines:
        line = line.split(",")
        actual_decisions = np.append(actual_decisions,int(line[10]))
        expert_d = np.empty(0).astype(int)
        expert_d = np.append(expert_d,int(line[7]))
        expert_d = np.append(expert_d,int(line[8]))
        expert_d = np.append(expert_d,int(line[9]))
        expert_decisions.append(expert_d)
        if int(line[6]) not in contextMap.keys():
            contextMap[int(line[6])] = contextCount
            contextCount += 1
        contexts = np.append(contexts,contextMap[int(line[6])])
    expert_decisions = np.array(expert_decisions)
    infile.close()
    w = WMA(3)    
    w.train(expert_decisions,actual_decisions)
    outfile = open(os.getcwd()+"\\..\\output\\output.csv","w")
    w.write(outfile)
    cw = CWMA(3,len(contextMap.keys()))
    cw.train(expert_decisions,actual_decisions,contexts)
    cw.write(outfile)
    outfile.close()
    
if __name__ == '__main__':
   main()