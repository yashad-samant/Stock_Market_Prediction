# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:53:09 2018

@author: Yashad
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sk
import padasip as pa
import csv
import time


# Extractig data using pandas
data = []
f = open('AAPL.csv', 'r')
reader = csv.reader(f)
for row in reader:
    data.append(row) 
f.close()
data = np.asarray(data)
names = data[0, 1:6]
data = data[1:,1:6]
data = data.astype(float)
n,d = data.shape


#Standardize the data
means = data.mean(0)
stds = data.std(0)
n,d = data.shape
data = np.insert( (data - means)/stds, 0, 1, axis=1)


#Insight on relevant data
plt.figure(figsize=(10,10))
for i in range(d):   
    plt.subplot(2,3,i+1)
    plt.plot(data[:,i])
    plt.ylabel('magnitude')
    plt.xlabel('samples')
    plt.title(names[i])

# train, test & validate
rowIndices = np.arange(len(data))
np.random.shuffle(rowIndices)

train = data[0:int(0.5*len(rowIndices)),:]
test = data[int(0.5*len(rowIndices)):int(0.75*len(rowIndices)),:]
validate = data[int(0.75*len(rowIndices)):int(len(rowIndices)),:]

# Generating time series
start_time_train = time.time()
x = []
N = 3
for i in range(N+1):
    x.append(train[N+1-i:len(train)-1-i,:])
    
# LMS algorithm

def LMS (d, x, mu):
    len_x = len(x)
    
    # create empty arrays
    y = np.zeros(d.shape)
    e = np.zeros(d.shape)
    w = [0.0,0.0,0.0]
    print(len(w))
    w_history = w
    dw = np.array(w)
    
    # adaptation loop   
    for i in range(d.shape[0]):
        temp = []
        for j in range(len_x):
            temp.append(np.dot(x[j][d.shape[0]-1-i], w[j]))

        y[i] = sum(temp)
        e[i] = d[i] - y[i]
        #print(e[i])
        #print(dw)
        for j in range(len_x):
            dw[j] = (mu*np.dot(x[j][d.shape[0]-1-i],e[i]))
        w += dw
        w_history.append(w)
    return y, e, w_history,w  

start_time_train = time.time()
[y, e, w_history,w] = LMS (x[0],x[1:4],0.01)
mse_train= sk.mean_squared_error(x[0],y)
end_time_train = time.time()
time_train = end_time_train-start_time_train

# plot trained output
plt.figure(figsize=(10,10))
for i in range(d):   
    plt.subplot(2,3,i+1)
    plt.plot(y[:,i])
    plt.ylabel('magnitude')
    plt.xlabel('samples')
    plt.title(names[i])


# Validation scheme
start_time_val = time.time()
xval = []
N = 3
for i in range(N+1):
    xval.append(validate[N+1-i:len(validate)-1-i,:])
y_cap = np.zeros(xval[0].shape)
for j in range(len(xval[1:4])):    
    y_cap += w[j]*xval[j]

mse_val = sk.mean_squared_error(xval[0],y_cap)
end_time_val = time.time()
time_val = end_time_val-start_time_val
plt.figure(figsize=(10,10))
for i in range(d):   
    plt.subplot(2,3,i+1)
    plt.plot(y_cap[:,i])
    plt.ylabel('magnitude')
    plt.xlabel('samples')
    plt.title(names[i])

# test scheme
start_time_test = time.time()
xtest = []
N = 3
for i in range(N+1):
    xtest.append(test[N+1-i:len(test)-1-i,:])
y_cap = np.zeros(xtest[0].shape)
for j in range(len(xtest[1:4])):    
    y_cap += w[j]*xtest[j]

mse_test = sk.mean_squared_error(xtest[0],y_cap)
end_time_test = time.time()
time_test = end_time_test-start_time_test
plt.figure(figsize=(10,10))
for i in range(d):   
    plt.subplot(2,3,i+1)
    plt.plot(data[:,i])
    plt.ylabel('magnitude')
    plt.xlabel('samples')
    plt.title(names[i])

#time plot

objects = ('time_train','time_test','time_val')
y_pos = np.arange(len(objects))
performance = [time_train, time_test, time_val]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('data segment')
plt.ylabel('time')
plt.title('Time Utilization')
 
plt.show()

# MSE plot
objects = ('time_train','time_test','time_val')
y_pos = np.arange(len(objects))
performance = [mse_train, mse_test, mse_val]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('data segment')
plt.ylabel('MSE')
plt.title('ERROR')
 
plt.show()

#Error Distribution
plt.hist(e)
plt.title('Error Distribution')
plt.xlabel('error')
plt.ylabel('samples')
plt.legend(['Open','High','Low','Close','Adj_Close'])
#RLS algorithm

# Comparison
#objects = ('time_train','time_test','time_val')
#y_pos = np.arange(len(objects))
#performance1 = [mse_train, mse_test, mse_val]
#performance2 = [0.06, 0.12, 0.53]
# 
#plt.bar(y_pos, performance1, align='center', alpha=0.5)
#plt.bar(y_pos, performance2, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.xlabel('data segment')
#plt.ylabel('MSE')
#plt.title('ERROR')
#plt.legend(['LMS','RLS'])
#plt.show()
