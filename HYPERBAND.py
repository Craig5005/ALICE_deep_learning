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

Building1 = os.getcwd() + '/Building1Electricity.csv'
Building1 = pd.read_csv(Building1)
Building1 = np.array(Building1)
Building1a = Building1[:,5:]
Building1aa = np.ravel(Building1a)
max_value_Building1aa = max(Building1aa)
for i in range(len(Building1aa)):
    Building1aa[i] = Building1aa[i] / max_value_Building1aa

Building2 = os.getcwd() + '/Building2Electricity.csv'
Building2 = pd.read_csv(Building2)
Building2 = np.array(Building2)
Building2a = Building2[:,5:]
Building2aa = np.ravel(Building2a)
max_value_Building2aa = max(Building2aa)
for i in range(len(Building2aa)):
    Building2aa[i] = Building2aa[i] / max_value_Building2aa

Building3 = os.getcwd() + '/Building3Electricity.csv'
Building3 = pd.read_csv(Building3)
Building3 = np.array(Building3)
Building3a = Building3[:,5:]
Building3aa = np.ravel(Building3a)
max_value_Building3aa = max(Building3aa)
for i in range(len(Building3aa)):
    Building3aa[i] = Building3aa[i] / max_value_Building3aa

Building1G = os.getcwd() + '/Building1Gas.csv'
Building1G = pd.read_csv(Building1G)
Building1G = np.array(Building1G)
Building1Ga = Building1G[:,5:]
Building1Ga = np.ravel(Building1Ga)
max_value_Building1Ga = max(Building1Ga)
for i in range(len(Building1Ga)):
    Building1Ga[i] = Building1Ga[i] / max_value_Building1Ga

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
    
dataset = np.append(Building1aa,Building1aa)
Dataset = 'Building 1 - Electricity'

run = 1

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
if Dataset == 'Temp':
    signal_length = 13
    print(Dataset)
if Dataset == 'data':
    signal_length = 100
    print(Dataset)
if Dataset == 'Building 1 - Electricity':
    signal_length = 320
    print(Dataset)
if Dataset == 'Building2':
    signal_length = 250
    print(Dataset)
if Dataset == 'Building3':
    signal_length = 330
    print(Dataset)
if Dataset == 'Building 1 - Gas':
    signal_length = 200
    print(Dataset)
if Dataset == 'Global_temp_data':
    signal_length = 100
    print(Dataset)
    
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

History_train = History_train_1 = History_train_2 = History_train_3 = np.array([])
History_validate = History_validate_1 = History_validate_2 = History_validate_3 = np.array([])
Predictions = np.array([])
Predictions_offline = np.array([])
DTW = np.array([])
Correlation = np.array([])
iters = len(dataset) / (signal_num*signal_length)

T = 3
count_1 = 0
count_2 = 0
count_3 = 0
gamma = 0.4

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
    
    OnlineFolder = BanditFolder+'/Action_1'+'/'
    if not os.path.exists(OnlineFolder):
        os.makedirs(OnlineFolder)   
    
    if count_1 == 0:
        model_10.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
    else:
        model_10.load_weights(OnlineFolder+'weights'+str(count_1-1)+'.h5', by_name=True)
    model_10.compile(loss='mean_squared_error', optimizer='adam')
    history_10 = model_10.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
    testPredict = model_10.predict(testX)
    History_train_1 = np.append(History_train_1, history_10.history['loss'])
    History_validate_1 = np.append(History_validate_1, history_10.history['val_loss'])
    model_10.save_weights(OnlineFolder+'weights'+str(count_1)+'.h5')
    count_1 = count_1 + 1
    
    if count_2 == 0:
        model_20.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
    else:
        model_20.load_weights(OnlineFolder+'weights'+str(count_2-1)+'.h5', by_name=True)
    model_20.compile(loss='mean_squared_error', optimizer='adam')
    history_20 = model_20.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
    testPredict = model_20.predict(testX)
    History_train_2 = np.append(History_train_2, history_20.history['loss'])
    History_validate_2 = np.append(History_validate_2, history_20.history['val_loss'])
    model_20.save_weights(OnlineFolder+'weights'+str(count_2)+'.h5')
    count_2 = count_2 + 1
    
    if count_3 == 0:
        model_30.load_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5', by_name=True)
    else:
        model_30.load_weights(OnlineFolder+'weights'+str(count_3-1)+'.h5')
    model_30.compile(loss='mean_squared_error', optimizer='adam')
    history_30 = model_30.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
    testPredict = model_30.predict(testX)
    History_train_3 = np.append(History_train_3, history_30.history['loss'])
    History_validate_3 = np.append(History_validate_3, history_30.history['val_loss'])
    model_30.save_weights(OnlineFolder+'weights'+str(count_3)+'.h5')
    count_3 = count_3 + 1
    
Arm_1 = np.cumsum(History_validate_1)
Arm_2 = np.cumsum(History_validate_2)
Arm_3 = np.cumsum(History_validate_3)

print 'After first stage reject:'
if max(Arm_1[-1],Arm_2[-1],Arm_3[-1]) == Arm_1[-1]:
    print 'Arm 1'
    n = 1
elif max(Arm_1[-1],Arm_2[-1],Arm_3[-1]) == Arm_2[-1]:
    print 'Arm 2'
    n = 2
elif max(Arm_1[-1],Arm_2[-1],Arm_3[-1]) == Arm_3[-1]:
    print 'Arm_3'
    n = 3
    
for t in range(3,12):
    print 'Iteration: %.1d' % (t), '----------------------------------------'
    context = dataset[t*signal_num*signal_length:(t+1)*signal_num*signal_length]
    test_size = signal_length
    train_size = signal_length
    train, test = context[:len(context)], context[:len(context)]
    validation_split = 0.5
    #reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    if n == 1:
        #model_10.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
        #model_10.compile(loss='mean_squared_error', optimizer='adam')
        #history_10 = model_10.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        #testPredict = model_10.predict(testX)
        #History_train_1 = np.append(History_train_1, history_10.history['loss'])
        #History_validate_1 = np.append(History_validate_1, history_10.history['val_loss'])
        #model_10.save_weights(OnlineFolder+'weights'+str(count_1)+'.h5')
        #count_1 = count_1 + 1
    
        model_20.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
        model_20.compile(loss='mean_squared_error', optimizer='adam')
        history_20 = model_20.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        testPredict = model_20.predict(testX)
        History_train_2 = np.append(History_train_2, history_20.history['loss'])
        History_validate_2 = np.append(History_validate_2, history_20.history['val_loss'])
        model_20.save_weights(OnlineFolder+'weights'+str(count_2)+'.h5')
        count_2 = count_2 + 1
    
        model_30.load_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5', by_name=True)
        model_30.compile(loss='mean_squared_error', optimizer='adam')
        history_30 = model_30.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        testPredict = model_30.predict(testX)
        History_train_3 = np.append(History_train_3, history_30.history['loss'])
        History_validate_3 = np.append(History_validate_3, history_30.history['val_loss'])
        model_30.save_weights(OnlineFolder+'weights'+str(count_3)+'.h5')
        count_3 = count_3 + 1
        
        #Arm_1 = np.cumsum(History_validate_1)
        Arm_2 = np.cumsum(History_validate_2)
        Arm_3 = np.cumsum(History_validate_3)
    
    if n == 2:
        model_10.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
        model_10.compile(loss='mean_squared_error', optimizer='adam')
        history_10 = model_10.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        testPredict = model_10.predict(testX)
        History_train_1 = np.append(History_train_1, history_10.history['loss'])
        History_validate_1 = np.append(History_validate_1, history_10.history['val_loss'])
        model_10.save_weights(OnlineFolder+'weights'+str(count_1)+'.h5')
        count_1 = count_1 + 1
    
        #model_20.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
        #model_20.compile(loss='mean_squared_error', optimizer='adam')
        #history_20 = model_20.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        #testPredict = model_20.predict(testX)
        #History_train_2 = np.append(History_train_2, history_20.history['loss'])
        #History_validate_2 = np.append(History_validate_2, history_20.history['val_loss'])
        #model_20.save_weights(OnlineFolder+'weights'+str(count_2)+'.h5')
        #count_2 = count_2 + 1
    
        model_30.load_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5', by_name=True)
        model_30.compile(loss='mean_squared_error', optimizer='adam')
        history_30 = model_30.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        testPredict = model_30.predict(testX)
        History_train_3 = np.append(History_train_3, history_30.history['loss'])
        History_validate_3 = np.append(History_validate_3, history_30.history['val_loss'])
        model_30.save_weights(OnlineFolder+'weights'+str(count_3)+'.h5')
        count_3 = count_3 + 1
        
        Arm_1 = np.cumsum(History_validate_1)
        #Arm_2 = np.cumsum(History_validate_2)
        Arm_3 = np.cumsum(History_validate_3)
    
    if n == 3:
        model_10.load_weights(BanditFolder+'/'+str(model_1)+str(Dataset)+'.h5', by_name=True)
        model_10.compile(loss='mean_squared_error', optimizer='adam')
        history_10 = model_10.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        testPredict = model_10.predict(testX)
        History_train_1 = np.append(History_train_1, history_10.history['loss'])
        History_validate_1 = np.append(History_validate_1, history_10.history['val_loss'])
        model_10.save_weights(OnlineFolder+'weights'+str(count_1)+'.h5')
        count_1 = count_1 + 1
    
        model_20.load_weights(BanditFolder+'/'+str(model_2)+str(Dataset)+'.h5', by_name=True)
        model_20.compile(loss='mean_squared_error', optimizer='adam')
        history_20 = model_20.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        testPredict = model_20.predict(testX)
        History_train_2 = np.append(History_train_2, history_20.history['loss'])
        History_validate_2 = np.append(History_validate_2, history_20.history['val_loss'])
        model_20.save_weights(OnlineFolder+'weights'+str(count_2)+'.h5')
        count_2 = count_2 + 1
    
        #model_30.load_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5', by_name=True)
        #model_30.compile(loss='mean_squared_error', optimizer='adam')
        #history_30 = model_30.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=2)
        #testPredict = model_30.predict(testX)
        #History_train_3 = np.append(History_train_3, history_30.history['loss'])
        #History_validate_3 = np.append(History_validate_3, history_30.history['val_loss'])
        #model_30.save_weights(OnlineFolder+'weights'+str(count_3)+'.h5')
        #count_3 = count_3 + 1

        Arm_1 = np.cumsum(History_validate_1)
        Arm_2 = np.cumsum(History_validate_2)
        #Arm_3 = np.cumsum(History_validate_3)

np.savetxt('E1_1_10', Arm_1, delimiter=',')
np.savetxt('E1_2_10', Arm_2, delimiter=',')
np.savetxt('E1_3_10', Arm_3, delimiter=',')        

#plt.figure(figsize=(20,10))
#plt.plot(Arm_1)
#plt.plot(Arm_2)
#plt.plot(Arm_3)
#plt.xlabel('t (rounds)')
#plt.ylabel('Cumulative regret, R(T)')
#plt.legend(['Arm 1', 'Arm 2', 'Arm 3'])
#plt.show()


