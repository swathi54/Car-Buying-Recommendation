#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:41:11 2017

@author: SwatzMac
@Program: Building a binary and multi-linear classifier on Car Data 
          to build recommendations on car buying data
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


#This function is used in the function readExcel(...) defined further below
def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values

#This function is used in the function readExcel(...) defined further below
def readExcelRange(excelfile,sheetname="Training Data",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values;
    return values[startrow-1:endrow,startcol-1:endcol]

#This is the function you can actually use within your program.
#See manner of usage further below in the section "Prepare Data"
def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data


def writeExcelData(x,excelfile,sheetname,startrow,startcol):
     from pandas import DataFrame, ExcelWriter
     from openpyxl import load_workbook
     df=DataFrame(x)
     book = load_workbook(excelfile)
     writer = ExcelWriter(excelfile, engine='openpyxl') 
     writer.book = book
     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
     df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
     writer.save()
     writer.close()
 


def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names

excelfile= "/Users/SwatzMac/Documents/Study/Classes/Machine Learning, Statistics and Python/Python_Programs/Linear Classifiers/CarData/Car_Data.xlsx"
excelfile1 = "/Users/SwatzMac/Documents/Study/Classes/Machine Learning, Statistics and Python/Python_Programs/Linear Classifiers/CarData/CarDataResults.xlsx"


sheets=getSheetNames(excelfile)
#print(sheets)

Xinit=(readExcel(excelfile, #General method of calling readExcel
                  sheetname='TrainingData',
                  startrow=2,
                  endrow=1729,
                  startcol=1,
                  endcol=7));

#print (Xinit.shape)   
print (Xinit)       

# We can't do much with the labeled data as it exists, so we can use LabelEncoder()
# to turn the labels into unique integer values
#le = preprocessing.LabelEncoder()
#for col in Xinit:
#    Xinit[col] = le.fit_transform(Xinit[col])
#    
#print(Xinit)

#Kesler's construction on PRICE column
price = (readExcel(excelfile, #General method of calling readExcel
                  sheetname='TrainingData',
                  startrow=2,
                  endrow=1729,
                  startcol=1,
                  endcol=1))
print(price.shape)  

price2 = np.ones([len(price),4])
#print(price2)

#
## Initialize all values in T2 to -1
price2 = np.dot (-1,price2)
print(price2.shape)

# 0 = low, 1 = med, 2 = high, 3 = vhigh
for i in range(len(price)):
    label = price[i]
    index = -1
    if label == 'vhigh':
        index = 3
    elif label == 'high':
        index = 2
    elif label == 'med':
        index = 1
    elif label == 'low':
        index = 0
    else:
        index = -1
    
    price2[i][index] = 1
#print(price2)

#Kesler's construction on MAINTAINANCE 
maint = Xinit[:,1]
maint2 = np.ones([len(maint),4])

maint2 = np.dot (-1,maint2)

# 0 = low, 1 = med, 2 = high, 3 = vhigh
for i in range(len(maint)):
    label = maint[i]
    index = -1
    if label == 'vhigh':
        index = 3
    elif label == 'high':
        index = 2
    elif label == 'med':
        index = 1
    elif label == 'low':
        index = 0
    else:
        index = -1
    
    maint2[i][index] = 1
#print(maint2)


#Kesler's construction on DOORS 
door = Xinit[:,2]
door2 = np.ones([len(door),4])

door2 = np.dot (-1,door2)

# 0 = 2, 1 = 3, 2 = 4, 3 = 5doors
for i in range(len(door)):
    label = door[i]
    index = -1
    if label == 5:
        index = 3
    elif label == 4:
        index = 2
    elif label == 3:
        index = 1
    elif label == 2:
        index = 0
    else:
        index = -1
    
    door2[i][index] = 1
#print(door2)

#Kesler's construction on PERSONS 
person = Xinit[:,3]
person2 = np.ones([len(person),3])

person2 = np.dot (-1,person2)

# 0 = 2persons, 1 = 4persons, 2 = 5persons
for i in range(len(person)):
    label = person[i]
    index = -1
    if label == 5:
        index = 2
    elif label == 4:
        index = 1
    elif label == 2:
        index = 0
    else:
        index = -1
    
    person2[i][index] = 1
#print(person2)

#Kesler's construction on TRUNK 
trunk = Xinit[:,4]
trunk2 = np.ones([len(trunk),3])

trunk2 = np.dot (-1,trunk2)

# 0 = small, 1 = med, 2 = big
for i in range(len(trunk)):
    label = trunk[i]
    index = -1
    if label == 'big':
        index = 2
    elif label == 'med':
        index = 1
    elif label == 'small':
        index = 0
    else:
        index = -1
    
    trunk2[i][index] = 1
#print(trunk2)


#Kesler's construction on SAFETY 
safety = Xinit[:,5]
safety2 = np.ones([len(safety),3])

safety2 = np.dot (-1,safety2)

# 0 = low, 1 = med, 2 = high
for i in range(len(safety)):
    label = safety[i]
    index = -1
    if label == 'high':
        index = 2
    elif label == 'med':
        index = 1
    elif label == 'low':
        index = 0
    else:
        index = -1
    
    safety2[i][index] = 1
#print(safety2)

X = np.concatenate((price2,maint2,door2,person2,trunk2,safety2),axis = 1)
print(X.shape)
#writeExcelData(X,excelfile1,"Sheet1",1,1)

# Appending a column of 1s to the start of the matrix X = Xa
Xa = np.c_[np.ones(len(X)),X]
print (Xa.shape)


#Kesler's construction on Recommendation Label - Target T 
T = Xinit[:,6]
T_reco_given = np.ones([len(T),1])
T2 = np.ones([len(T),4])

T2 = np.dot (-1,T2)

# 0 = unacc, 1 = acc, 2 = good, 3 = vgood
for i in range(len(T)):
    label = T[i]
    index = -1
    if label == 'vgood':
        index = 3
    elif label == 'good':
        index = 2
    elif label == 'acc':
        index = 1
    elif label == 'unacc':
        index = 0
    else:
        index = -1
        
    T_reco_given[i] = index
    T2[i][index] = 1
#print(T2)

Xa_Inv = np.linalg.pinv(Xa)
print(Xa_Inv.shape)

### Multi-class Classifier 
W4 = np.dot(Xa_Inv,T2)
print(W4.shape)
print(W4)
#writeExcelData(W4,excelfile1,"Sheet2",1,1)

## Predicting the Recommendation LAbel 
T_reco = np.dot(Xa,W4)
print(T_reco)


# Making T_reco_result with -1's
#T_reco_result = np.ones([len(T_reco),1])
#T_reco_result = np.empty([len(T_reco),1], dtype=object)
#print(T_reco_result)

T_reco_result = []
T_reco_predicted = []
# Returns the index value of the highest feature vector from a array
for i in range(len(T_reco)):
    index = np.argmax(T_reco[i])
    if index == 0:
        label = 'unacc'
    elif index == 1:
        label = 'acc' 
    elif index == 2:
        label = 'good'  
    elif index == 3:
        label = 'vgood'  
    T_reco_result.append(label)
    T_reco_predicted.append(index)

T_reco_result = np.array(T_reco_result).reshape(len(T_reco_result),1)
T_reco_predicted = np.array(T_reco_predicted).reshape(len(T_reco_predicted),1)
#print(T_reco_result.shape)

Perf_reco = np.zeros([4,4])

row = 7
col = 7
for i in range(len(T_reco_given)):
    row = (int) (T_reco_given[i])
    col = (int) (T_reco_predicted[i])
    Perf_reco[row][col] += 1

print(Perf_reco)
#writeExcelData(Perf_reco,excelfile1,"Sheet3",1,1)


### BINARY CLASSIFIER ####
#Binary construction on Recommendation Label - Target T 
TBi = Xinit[:,6]
TBi_reco_given = np.ones([len(TBi),1])


# 0 = unacc, 1 = acc, 2 = good, 3 = vgood
for i in range(len(TBi)):
    label = TBi[i]
   
    if label == 'vgood':
        TBi_reco_given[i] = 1
    elif label == 'good':
        TBi_reco_given[i] = 1
    elif label == 'acc':
        TBi_reco_given[i] = 1
    elif label == 'unacc':
        TBi_reco_given[i] = -1
#    else:
#        TBi_reco_given[i] = -1
        

W = np.dot(Xa_Inv,TBi_reco_given)
print(W)
print(W.shape)

TBi_reco = np.dot(Xa,W)
TBi_reco_predicted = np.sign(TBi_reco)
#print(TBi_reco_predicted.shape)


# creating a 2 by 2 empty matrix for Recommendation Failure
Perf_binary_reco = np.zeros([2,2])


row = 2
col = 2
for i in range(len(TBi_reco_given)):
    if TBi_reco_given[i] < 0:
        row = 0
    else:
        row = 1
    if TBi_reco_predicted[i] < 0:
        col = 0
    else:
        col = 1
        
    Perf_binary_reco[row][col] += 1 

print(Perf_binary_reco)




