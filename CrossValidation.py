#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:35:37 2021

@author: huangjianxiang
"""

#from numpy import mean
#from numpy import std
#from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import pyreadstat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#creat dataset
#X = np.array([[1, 2], [2, 4],[5,6],[7,8]])
#y = np.array([1, 2, 3, 4])

def loocv(X,y):
    r2cv=[]
    rmsecv=[]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    model=linear_model.LinearRegression()
    
    model.fit(X,y)
    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #training set 
        y_predict=model.predict(X_train)
        #r2 cross validation
        r2cv.append(qloo2(y_train,y_predict))
        #rmse cross validation
        rmsecv.append(np.sqrt(mean_squared_error(y_test,model.predict(X_test))))
    return np.mean(np.array(r2cv)),np.mean(np.array(rmsecv))
    
    
def qloo2(y_experiment,y_predict):
    #determination coefficients
    up=0
    down=0
    y_ave=np.mean(y_experiment)
    for k in range(len(y_experiment)):
        up+=(y_experiment[k]-y_predict[k])**2
        down+=(y_experiment[k]-y_ave)**2
    return (1-up/down)


def gendata():
    #load local data
    df,meta=pyreadstat.read_sav("imine-retreatment-H.sav")
    df1,meta1=pyreadstat.read_sav("New-pKBHX-all-Cm-final.sav")
    #df.to_excel("imine-retreatment-H.xlsx")
    #df1.to_excel("New-pKBHX-all-Cm-final.xlsx")
    return np.array(df1),np.array(df)


def extractdata(data,col,x_name,data2=False):
    #a dictionary to store the discriptor names
    #split data
    if not data2:
        dict_list={0:"pKBHX", 1:"pKBHX_1", 2:"pKBHX_2",3: "site_X", 4:"site_C", 5:"site_S", 6:"site_O", 7:"site_O_1", 8:"site_N",9:"site_N_1",10: "site_N_2",11: "site_N_primary", 12:"site_N_second", 13:"site_N_tert", 14:"site_N_aromatic", 15:"site_N_imine", 16:"site_N_nitrile"}
        for i in range(1,35):
            if i<10:
                dict_list[i+16]="esp0"+str(i)
            else:
                dict_list[i+16]="esp"+str(i)
    
    
    if data2:
        dict_list={0:"pKBHX", 1:"pKBHX_1", 2:"pKBHX_2"}
        for i in range(1,35):
            if i<10:
                dict_list[i+2]="esp0"+str(i)
            else:
                dict_list[i+2]="esp"+str(i)
#    print(dict_list)
    #inverse the name dictionary 
    inverse_dic={}
    for key,val in dict_list.items():
        inverse_dic[val]=key
    
    def retrieve(list_in):
        list_out=[]
        for j in range(len(list_in)):
            list_out.append(inverse_dic[list_in[j]])
        return list_out
    
    ################  ALL  ###########################
    #x_name=['esp17','esp16','esp03']
    x_value=retrieve(x_name)
    
    #transpose data and take the values and then transpose
    #data1=data.T
    #data1[X].T
    data1= data.copy()
    Y=data1[:,col]
    X=data1.T[x_value].T
    
    X1=X[np.logical_not(np.tile(np.isnan(Y),(len(x_name),1)).T)]
    
    Y1=Y[np.logical_not(np.isnan(Y))]
    
    X2=X1.reshape(np.size(X1)//len(x_name),len(x_name))
    
    return Y1,X2



def mccv(X,y):
    
    qext2=[]
    #monte carlo cross validation for 2^16=65536 times
    for i in range(65536):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        #model part
        model=linear_model.LinearRegression()
        
        model.fit(X,y)
        
        y_predict=model.predict(X_train)
        
        #calculate qext2
        qext2.append(qloo2(y_train,y_predict))
    #convert to numpy array
    test=np.array(qext2)
    #calculate the probability into 100 bins
    hist,_=np.histogram(test,bins=np.linspace(np.min(test),np.max(test),num=100))
    
    #normalize the probability
    pcum=np.cumsum(hist)/65536
    
    #plot the data for checking purposes
    plt.plot(_[1:],np.cumsum(hist)/65536,label='Pcum')
    plt.plot(_[1:],hist/np.max(hist),label='Probability density')
    plt.plot(_[1:],1-pcum,label='1-Pcum')
    plt.legend()
    
    #median qext2
    for i in range(np.size(pcum)):
        if pcum[i]<=0.5 and pcum[i+1]>0.5:
#            print(i)
            median=(_[1:][i]+_[1:][i+1])/2
        else:
            pass
        
    #peak qext2
    max_id=np.where(hist==np.max(hist))  
    peak=np.float(_[1:][max_id])
    
    #integral qext2
    integral=np.trapz((1-pcum),dx=(_[1]-_[0])) + _[1]
    return median,peak,integral
    

data1,data2=gendata()
y,x=extractdata(data1,2,['esp02','esp03','esp09','esp17'])

print(mccv(x,y))