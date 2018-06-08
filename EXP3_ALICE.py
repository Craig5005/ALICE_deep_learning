import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#%matplotlib inline

#0000FF#0000FF

ec2_cpu_1 = os.getcwd() + '/ec2_cpu_1.csv'
ec2_cpu_1 = pd.read_csv(ec2_cpu_1)
ec2_cpu_1 = np.array(ec2_cpu_1)
ec2_cpu_1 = ec2_cpu_1[:,1]
ec2_cpu_1 = ec2_cpu_1.astype(np.float)

ec2_cpu_2 = os.getcwd() + '/ec2_cpu_2.csv'
ec2_cpu_2 = pd.read_csv(ec2_cpu_2)
ec2_cpu_2 = np.array(ec2_cpu_2)
ec2_cpu_2 = ec2_cpu_2[:,1]
ec2_cpu_2 = ec2_cpu_2.astype(np.float)

disease_person_1_data = os.getcwd() + '/disease_person1.csv'
disease_person1 = pd.read_csv(disease_person_1_data)
disease_person_1 = np.array(disease_person1)
disease_person_1 = disease_person_1[2:,:]
disease_person_1 = disease_person_1[:,1]
disease_person_1 = disease_person_1.astype(np.float)

max_value_disease_person_1 = max(disease_person_1)
for i in range(len(disease_person_1)):
    disease_person_1[i] = disease_person_1[i] / max_value_disease_person_1

disease_person_2_data = os.getcwd() + '/disease_person2.csv'
disease_person2 = pd.read_csv(disease_person_2_data)
disease_person_2 = np.array(disease_person2)
disease_person_2 = disease_person_2[2:,:]
disease_person_2 = disease_person_2[:,1]
disease_person_2 = disease_person_2.astype(np.float)

max_value_disease_person_2 = max(disease_person_2)
for i in range(len(disease_person_2)):
    disease_person_2[i] = disease_person_2[i] / max_value_disease_person_2

disease_person_3_data = os.getcwd() + '/disease_person3.csv'
disease_person3 = pd.read_csv(disease_person_3_data)
disease_person_3 = np.array(disease_person3)
disease_person_3 = disease_person_3[2:,:]
disease_person_3 = disease_person_3[:,1]
disease_person_3 = disease_person_3.astype(np.float)

max_value_disease_person_3 = max(disease_person_3)
for i in range(len(disease_person_3)):
    disease_person_3[i] = disease_person_3[i] / max_value_disease_person_3

healthy_person_1_data = os.getcwd() + '/healthy_person1.csv'
healthy_person1 = pd.read_csv(healthy_person_1_data)
healthy_person_1 = np.array(healthy_person1)
healthy_person_1 = healthy_person_1[2:,:]
healthy_person_1 = healthy_person_1[:,1]
healthy_person_1 = healthy_person_1.astype(np.float)

max_value_healthy_person_1 = max(healthy_person_1)
for i in range(len(healthy_person_1)):
    healthy_person_1[i] = healthy_person_1[i] / max_value_healthy_person_1

healthy_person_2_data = os.getcwd() + '/healthy_person2.csv'
healthy_person2 = pd.read_csv(healthy_person_2_data)
healthy_person_2 = np.array(healthy_person2)
healthy_person_2 = healthy_person_2[2:,:]
healthy_person_2 = healthy_person_2[:,1]
healthy_person_2 = healthy_person_2.astype(np.float)

max_value_healthy_person_2 = max(healthy_person_2)
for i in range(len(healthy_person_2)):
    healthy_person_2[i] = healthy_person_2[i] / max_value_healthy_person_2

healthy_person_3_data = os.getcwd() + '/healthy_person3.csv'
healthy_person3 = pd.read_csv(healthy_person_3_data)
healthy_person_3 = np.array(healthy_person3)
healthy_person_3 = healthy_person_3[2:,:]
healthy_person_3 = healthy_person_3[:,1]
healthy_person_3 = healthy_person_3.astype(np.float)

max_value_healthy_person_3 = max(healthy_person_3)
for i in range(len(healthy_person_3)):
    healthy_person_3[i] = healthy_person_3[i] / max_value_healthy_person_3

healthy_person_4_data = os.getcwd() + '/healthy_person4.csv'
healthy_person4 = pd.read_csv(healthy_person_4_data)
healthy_person_4 = np.array(healthy_person4)
healthy_person_4 = healthy_person_4[2:,:]
healthy_person_4 = healthy_person_4[:,1]
healthy_person_4 = healthy_person_4.astype(np.float)

max_value_healthy_person_4 = max(healthy_person_4)
for i in range(len(healthy_person_4)):
    healthy_person_4[i] = healthy_person_4[i] / max_value_healthy_person_4

E_1 = os.getcwd() + '/Building1Electricity.csv'
E_1 = pd.read_csv(E_1)
E_1 = np.array(E_1)
E_1 = E_1[:,5:]
E_1 = np.ravel(E_1)
E_1 = E_1.astype(np.float)
E_1 = E_1[100:]
max_value_E_1 = max(E_1)
for i in range(len(E_1)):
    E_1[i] = E_1[i] / max_value_E_1
    
E_3 = os.getcwd() + '/Building3Electricity.csv'
E_3 = pd.read_csv(E_3)
E_3 = np.array(E_3)
E_3 = E_3[:,5:]
E_3 = np.ravel(E_3)
E_3 = E_3.astype(np.float)
E_3 = E_3[80:]
max_value_E_3 = max(E_3)
for i in range(len(E_3)):
    E_3[i] = E_3[i] / max_value_E_3
    
dataset = np.append(healthy_person_1,healthy_person_1)
Dataset = 'healthy_person_1'

run = 4

signal_num = 2
if Dataset == 'healthy_person_1':
    signal_length = 160
    print(Dataset)
if Dataset == 'healthy_person_2':
    signal_length = 190
    print(Dataset)
if Dataset == 'healthy_person_3':
    signal_length = 130
    print(Dataset)
if Dataset == 'healthy_person_4':
    signal_length = 170
    print(Dataset)
if Dataset == 'disease_person_1':
    signal_length = 160
    print(Dataset) 
if Dataset == 'disease_person_2':
    signal_length = 160
    print(Dataset) 
if Dataset == 'disease_person_3':
    signal_length = 148
    print(Dataset)
if Dataset == 'Sine wave':
    signal_length = 21
    print(Dataset)
if Dataset == 'cpu_1':
    signal_length = 150
    print(Dataset)
if Dataset == 'cpu_2':
    signal_length = 10
    print(Dataset)
if Dataset == 'E_1':
	signal_length = 300
if Dataset == 'E_3':
	signal_length = 150
	
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

look_back = 5

K = 3

import shutil
def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
remove_folder('./ALICE_BANDIT/'+str(Dataset)+'/RUN'+str(run))

# create the networks
# create and fit the network
model_1 = Sequential()
model_1.add(Dense(10, input_dim=look_back, activation='relu', name='dense_1_offline'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1))
#model_1.compile(loss='mean_squared_error', optimizer='adam')

model_2 = Sequential()
model_2.add(Dense(10, input_dim=look_back, activation='relu', name='dense_2_offline'))
model_2.add(Dense(100, activation='relu'))
#model_2.add(Dense(10, activation='relu'))
model_2.add(Dense(1))
#model_2.compile(loss='mean_squared_error', optimizer='adam')

model_3 = Sequential()
model_3.add(Dense(10, input_dim=look_back, activation='relu', name='dense_3_offline'))
for _ in range(50):
    model_3.add(Dense(3, activation='relu'))
model_3.add(Dense(1))
#model_3.compile(loss='mean_squared_error', optimizer='adam')

model_10 = model_11 = model_12 = model_1
model_20 = model_21 = model_22 = model_2
model_30 = model_31 = model_32 = model_3

BanditFolder = './ALICE_BANDIT/'+str(Dataset)+'/RUN'+str(run)
if not os.path.exists(BanditFolder):
    os.makedirs(BanditFolder)
model_10.save_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5')
model_11.save_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5')
model_12.save_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5')
model_20.save_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5')
model_21.save_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5')
model_22.save_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5')
model_30.save_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5')
model_31.save_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5')
model_32.save_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5')

History_train = np.array([])
History_validate = np.array([])
Predictions = np.array([])
Predictions_offline = np.array([])
DTW = np.array([])
Correlation = np.array([])
iters = len(dataset) / (signal_num*signal_length)

T = 12
count_1 = 0
count_2 = 0
count_3 = 0
gamma = 0.4

np.random.seed(7)
probs = np.random.rand(T)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

weights = np.ones(K)
P = weights / K
Probs = np.zeros([T,K])
Weights = np.zeros([T,K])
Arms = np.array([])
D = 3

for t in range(T):
    print 'Iteration: %.1d' % (t), '----------------------------------------'
    context = dataset[t*signal_num*signal_length:(t+1)*signal_num*signal_length]
    test_size = signal_length
    train_size = signal_length
    train, test = context[:len(context)], context[:len(context)]
    validation_split = 0.5
    #reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    p = probs[t]
    #p = np.random.uniform()
    
    if p >= 0 and p < P[0]:
        print p
        print('ARM 1')
        arm = 1
        Arms = np.append(Arms,arm)
        OnlineFolder = BanditFolder+'/Action_1'+'/'
        if not os.path.exists(OnlineFolder):
            os.makedirs(OnlineFolder)   
        if count_1 == 0:
            model_10.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
            model_11.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
            model_12.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
        else:
            model_10.load_weights(OnlineFolder+'weights'+str(count_1-1)+'.h5', by_name=True)
            model_11.load_weights(OnlineFolder+'weights'+str(count_1-1)+'.h5', by_name=True)
            model_12.load_weights(OnlineFolder+'weights'+str(count_1-1)+'.h5', by_name=True)
        model_10.compile(loss='mean_squared_error', optimizer='adam')
        history_10 = model_10.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        model_11.compile(loss='mean_squared_error', optimizer='adam')
        history_11 = model_11.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        model_12.compile(loss='mean_squared_error', optimizer='adam')
        history_12 = model_11.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        if history_10.history['val_loss'][0] == min(history_10.history['val_loss'][0],history_11.history['val_loss'][0],history_12.history['val_loss'][0]):
            model_1 = model_10
            history_1 = history_10
        elif history_11.history['val_loss'][0] == min(history_10.history['val_loss'][0],history_11.history['val_loss'][0],history_12.history['val_loss'][0]):
            model_1 = model_11
            history_1 = history_11
        elif history_12.history['val_loss'][0] == min(history_10.history['val_loss'][0],history_11.history['val_loss'][0],history_12.history['val_loss'][0]):
            model_1 = model_12
            history_1 = history_12
        model_1.save_weights(OnlineFolder+'weights'+str(count_1)+'.h5')
        count_1 = count_1 + 1
        reward1 = history_1.history['val_loss']
        reward1 = 1 - reward1[0]
        reward1_est = reward1/P[0]
        P[0] = (1-gamma)*(weights[0] / sum(weights))+(gamma)/np.float(K)
        P[1] = (1-gamma)*(weights[1] / sum(weights))+(gamma)/np.float(K)
        P[2] = (1-gamma)*(weights[2] / sum(weights))+(gamma)/np.float(K)
        weights[0] = weights[0]*np.exp((gamma/np.float(K))*reward1_est)
        if weights[0] > 99999:
            weights[0] = 99999
        Weights[t,:] = weights
        Probs[t,:] = P
        print P
        testPredict = model_1.predict(testX)
        History_train = np.append(History_train, history_1.history['loss'])
        History_validate = np.append(History_validate, history_1.history['val_loss'])
        Predictions = np.append(Predictions,testPredict)
        corr = np.correlate(testX[:,look_back-1],testPredict[:,0])
        Correlation = np.append(Correlation,corr)
        print("Saving model...")
        model_1.save(OnlineFolder+'model_1_'+str(count_1)+'.h5')
        del model_1
        print("Model saved to disk.")
        model_1 = load_model(OnlineFolder+'model_1_'+str(count_1)+'.h5')
    elif p >= P[0] and p < (P[0] + P[1]):
        print p
        print('ARM 2')
        arm = 2
        Arms = np.append(Arms,arm)
        OnlineFolder = BanditFolder+'/Action_2'+'/'
        if not os.path.exists(OnlineFolder):
            os.makedirs(OnlineFolder)   
        if count_2 == 0:
            model_20.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
            model_21.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
            model_22.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
        else:
            model_20.load_weights(OnlineFolder+'weights'+str(count_2-1)+'.h5', by_name=True)
            model_21.load_weights(OnlineFolder+'weights'+str(count_2-1)+'.h5', by_name=True)
            model_22.load_weights(OnlineFolder+'weights'+str(count_2-1)+'.h5', by_name=True)
        model_20.compile(loss='mean_squared_error', optimizer='adam')
        history_20 = model_20.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        model_21.compile(loss='mean_squared_error', optimizer='adam')
        history_21 = model_21.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        model_22.compile(loss='mean_squared_error', optimizer='adam')
        history_22 = model_22.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        if history_20.history['val_loss'][0] == min(history_20.history['val_loss'][0],history_21.history['val_loss'][0],history_22.history['val_loss'][0]):
            model_2 = model_20
            history_2 = history_20
        elif history_21.history['val_loss'][0] == min(history_20.history['val_loss'][0],history_21.history['val_loss'][0],history_22.history['val_loss'][0]):
            model_2 = model_21
            history_2 = history_21
        elif history_22.history['val_loss'][0] == min(history_20.history['val_loss'][0],history_21.history['val_loss'][0],history_22.history['val_loss'][0]):
            model_2 = model_22
            history_2 = history_22
        model_2.save_weights(OnlineFolder+'weights'+str(count_2)+'.h5')
        count_2 = count_2 + 1
        reward2 = history_2.history['val_loss']
        reward2 = 1 - reward2[0]
        reward2_est = reward2/P[1]
        P[0] = (1-gamma)*(weights[0] / sum(weights))+(gamma)/np.float(K)
        P[1] = (1-gamma)*(weights[1] / sum(weights))+(gamma)/np.float(K)
        P[2] = (1-gamma)*(weights[2] / sum(weights))+(gamma)/np.float(K)
        Probs[t,:] = P
        weights[1] = weights[1]*np.exp((gamma/np.float(K))*reward2_est)
        if weights[1] > 99999:
            weights[1] = 99999
        Weights[t,:] = weights
        print P
        testPredict = model_2.predict(testX)
        History_train = np.append(History_train, history_2.history['loss'])
        History_validate = np.append(History_validate, history_2.history['val_loss'])
        Predictions = np.append(Predictions,testPredict)
        corr = np.correlate(testX[:,look_back-1],testPredict[:,0])
        Correlation = np.append(Correlation,corr)
        print("Saving model...")
        model_2.save(OnlineFolder+'model_2_'+str(count_2)+'.h5')
        del model_2
        print("Model saved to disk.")
        model_2 = load_model(OnlineFolder+'model_2_'+str(count_2)+'.h5')
    elif p >= (P[1] + P[2]) and p <= 1:
        print p
        print('ARM 3')
        arm = 3
        Arms = np.append(Arms,arm)
        OnlineFolder = BanditFolder+'/Action_3'+'/'
        if not os.path.exists(OnlineFolder):
            os.makedirs(OnlineFolder)   
        if count_3 == 0:
            model_3.load_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5', by_name=True)
        else:
            model_3.load_weights(OnlineFolder+'weights'+str(count_3-1)+'.h5', by_name=True)
        model_30.compile(loss='mean_squared_error', optimizer='adam')
        history_30 = model_30.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        model_31.compile(loss='mean_squared_error', optimizer='adam')
        history_31 = model_31.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        model_32.compile(loss='mean_squared_error', optimizer='adam')
        history_32 = model_32.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        if history_30.history['val_loss'][0] == min(history_30.history['val_loss'][0],history_31.history['val_loss'][0],history_32.history['val_loss'][0]):
            model_3 = model_30
            history_3 = history_30
        elif history_31.history['val_loss'][0] == min(history_30.history['val_loss'][0],history_31.history['val_loss'][0],history_32.history['val_loss'][0]):
            model_3 = model_31
            history_3 = history_31
        elif history_32.history['val_loss'][0] == min(history_30.history['val_loss'][0],history_31.history['val_loss'][0],history_32.history['val_loss'][0]):
            model_3 = model_32
            history_3 = history_32
        model_3.save_weights(OnlineFolder+'weights'+str(count_3)+'.h5')
        count_3 = count_3 + 1
        reward3 = history_3.history['val_loss']
        reward3 = 1 - reward3[0]
        reward3_est = reward3/P[2]
        P[0] = (1-gamma)*(weights[0] / sum(weights))+(gamma)/np.float(K)
        P[1] = (1-gamma)*(weights[1] / sum(weights))+(gamma)/np.float(K)
        P[2] = (1-gamma)*(weights[2] / sum(weights))+(gamma)/np.float(K)
        Probs[t,:] = P
        Weights[t,:] = weights
        weights[2] = weights[2]*np.exp((gamma/np.float(K))*reward3_est)
        if weights[2] > 99999:
            weights[2] = 99999
        print P
        testPredict = model_3.predict(testX)
        History_train = np.append(History_train, history_3.history['loss'])
        History_validate = np.append(History_validate, history_3.history['val_loss'])
        Predictions = np.append(Predictions,testPredict)
        corr = np.correlate(testX[:,look_back-1],testPredict[:,0])
        Correlation = np.append(Correlation,corr)
        print("Saving model...")
        model_3.save(OnlineFolder+'model_3_'+str(count_3)+'.h5')
        del model_3
        print("Model saved to disk.")
        model_3 = load_model(OnlineFolder+'model_3_'+str(count_3)+'.h5')
        
#f, ax = plt.subplots(4, 1, figsize=(20,8))
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                wspace=None, hspace=0.4)

#Anomalies = np.zeros(len(Correlation))
#Anomalies[:] = np.nan
#x = np.linspace(0, len(Correlation), num=len(Correlation))
#Avg = np.zeros(len(Correlation))
#Upper = np.zeros(len(Correlation))
#Lower = np.zeros(len(Correlation))
#Avg[:] = np.median(Correlation)
#Upper[:] = np.percentile(Correlation, 90)
#Lower[:] = np.percentile(Correlation, 10)
#for i in range(len(Correlation)):
#    if Correlation[i]<Lower[i] or Correlation[i]>Upper[i]:
#        Anomalies[i] = Correlation[i]

#buffers = np.zeros(signal_num*signal_length)
#buffers[:] = np.nan
#dataset1 = np.append(buffers,dataset)
#Predictions1 = np.append(buffers,Predictions)
#ax[0].plot(Predictions, 'k', linewidth=4)
#ax[0].plot(dataset[:len(Predictions)], 'y')
#ax[0].set_ylabel('mV')
#ax[0].set_title('Output predictions')

#ax[1].plot(Probs, linewidth=3)
#ax[1].set_title('Arm importance weighting')
#ax[1].set_ylim([-0.1,1.1])

#ax[2].plot(Correlation[1:], 'k', linewidth=4)
#ax[2].plot(Upper, 'r--', linewidth=2)
#ax[2].plot(Lower, 'r--', linewidth=2)
#ax[2].fill_between(x,Upper,Lower, color='grey')
#ax[2].plot(Anomalies[1:], 'ro', markersize=10)
#ax[2].set_title('Anomaly prediction')

#ax[3].plot(History_train, 'b')
#ax[3].plot(History_validate, 'r')
#ax[3].set_ylabel('MSE')
#ax[3].set_title('Model optimisation')

#plt.savefig(BanditFolder+'/Results'+str(run)+'.png')

EXP3_n_0 = np.cumsum(History_validate)

#np.savetxt('EXP3_E3_2', EXP3_n_0, delimiter=',')
np.savetxt('EXP3_P6', Probs, delimiter=',')
