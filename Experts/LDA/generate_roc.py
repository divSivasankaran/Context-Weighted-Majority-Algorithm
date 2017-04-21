# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:19:25 2017

@author: e0013178
"""
import matplotlib.pyplot as plt
import os
from math import log
path = os.getcwd() + "/../../FusionMethods/output/roc.csv"
file = open(path,"r")
lines = file.readlines()

total_genuine = 599
total_imposter = 540

wma_far = []
wma_frr = []
cwma_far = []
cwma_frr = []
prod_far = []
prod_frr = []
sum_far = []
sum_frr = []
max_far = []
max_frr = []
wms_far = []
wms_frr = []
cwms_far = []
cwms_frr = []

for line in lines:
    line = line.split(",")
    sum_far.append(float(line[0])/total_imposter)
    sum_frr.append((float(line[1])/total_genuine))
    max_far.append(float(line[2])/total_imposter)
    max_frr.append((float(line[3])/total_genuine))
    prod_far.append(float(line[4])/total_imposter)
    prod_frr.append((float(line[5])/total_genuine))
    wma_far.append(float(line[6])/total_imposter)
    wma_frr.append((float(line[7])/total_genuine))
    cwma_far.append(float(line[8])/total_imposter)
    cwma_frr.append((float(line[9])/total_genuine))
    wms_far.append(float(line[10])/total_imposter)
    wms_frr.append((float(line[11])/total_genuine))
    cwms_far.append(float(line[12])/total_imposter)
    cwms_frr.append(float(line[13])/total_genuine)
   
   

plt.plot(wma_far,wma_frr,label="WMA")
plt.plot(cwma_far,cwma_frr,label = "CWMA")
plt.plot(prod_far,prod_frr, label = "Product")
plt.plot(sum_far,sum_frr,label = "Sum")
plt.plot(max_far,max_frr, label = 'Max')

plt.plot(wms_far,wms_frr,label="WMS")
plt.plot(cwms_far,cwms_frr,label = "CWMS")
plt.legend()
plt.axis((0.0,0.1,0,0.1))
#plot.('log')
plt.show()