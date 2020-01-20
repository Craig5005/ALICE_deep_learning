# functions.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import shutil
import time
import os
from multiprocessing import Pool
from keras.models import Sequential, load_model
from keras.layers import Dense

def remove_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
		
def run_arms(arms):
	os.system('python {}'.format(arms))
	
def process_data(dataset):
	data = os.getcwd() + '/' + str(dataset) + '.csv'
	data = pd.read_csv(data)
	data = np.array(data)
	data = data[2:,:]
	data = data[:,1]
	data = data.astype(np.float)
	return data

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i+look_back])
	return np.array(dataX), np.array(dataY)
	
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
   
def RunningMedian(seq, M):
    seq = iter(seq)
    s = []   
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq,M)]    
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes    
    median = lambda : s[m] if bool(M&1) else (s[m-1]+s[m])*0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()    
    medians = [median()]   

    # Now slide the window by one point to the right for each new position (each pass through 
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()          # pop oldest from left
        d.append(item)             # push newest in from right
        del s[bisect_left(s, old)] # locate insertion point and then remove old 
        insort(s, item)            # insert newest such that new sort is not required        
        medians.append(median())  
    return medians

def expMa(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def load_file(item):
    return np.loadtxt(item, delimiter=',')[0], np.loadtxt(item, delimiter=',')[1]

def load_datasets(datasets):
	data_stream = np.array([])
	T1 = 0
	for i in range(len(datasets)):
		_data = process_data(datasets[i])
		signal_length = 160
		Total = len(_data) / signal_length
		for j in range(T1,T1+Total):
			np.savetxt('./ALICE_BANDIT/Datasets/Data'+str(j), _data[(j-T1)*signal_length : (j-T1)*signal_length + 2*signal_length])
		T1 = T1 + Total
		data_stream = np.append(data_stream, _data)
	return data_stream






#def load_datasets(datasets):
#    T1 = 0
#    data_stream = np.array([])
#    for i in range(len(datasets)):
#        _data = process_data(datasets[i])
#        signal_length = 160
#        Total = len(_data) / signal_length
#        data_stream = np.append(data_stream, _data)
#        for j in range(T1,T1+Total):
#			np.savetxt('./ALICE_BANDIT/Datasets/Data'+str(j), _data[(j-(T1))*signal_length : (j-(T1))*signal_length + 2*signal_length])
#		T1 = T1 + Total
#    return data_stream
