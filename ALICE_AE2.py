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
from matplotlib import animation, rc
from IPython.display import HTML

#%matplotlib inline

#0000FF#0000FF

#ec2_cpu_1 = os.getcwd() + '/ec2_cpu_1.csv'
#ec2_cpu_1 = pd.read_csv(ec2_cpu_1)
#ec2_cpu_1 = np.array(ec2_cpu_1)
#ec2_cpu_1 = ec2_cpu_1[:,1]
#ec2_cpu_1 = ec2_cpu_1.astype(np.float)

#ec2_cpu_2 = os.getcwd() + '/ec2_cpu_2.csv'
#ec2_cpu_2 = pd.read_csv(ec2_cpu_2)
#ec2_cpu_2 = np.array(ec2_cpu_2)
#ec2_cpu_2 = ec2_cpu_2[:,1]
#ec2_cpu_2 = ec2_cpu_2.astype(np.float)

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
    
dataset = disease_person_1
Dataset = 'disease_person_1'

run = 0

signal_num = 2

signal_length_h1 = 160
signal_length_h2 = 190
signal_length_h3 = 130
signal_length_h4 = 170

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
    
num_training_samples_h1 = int(len(healthy_person_1) / signal_length_h1)
Dataset_h1 = healthy_person_1[:num_training_samples_h1 * signal_length_h1]
Training_ecg_h1 = np.reshape(Dataset_h1, (-1, signal_length_h1))

#num_training_samples_h2 = int(len(healthy_person_2) / signal_length_h2)
#Dataset_h2 = healthy_person_2[:num_training_samples_h2 * signal_length_h2]
#Training_ecg_h2 = np.reshape(Dataset_h2, (-1, signal_length_h2))

#num_training_samples_h3 = int(len(healthy_person_3) / signal_length_h3)
#Dataset_h3 = healthy_person_3[:num_training_samples_h3 * signal_length_h3]
#Training_ecg_h3 = np.reshape(Dataset_h3, (-1, signal_length_h3))

#num_training_samples_h4 = int(len(healthy_person_4) / signal_length_h4)
#Dataset_h4 = healthy_person_4[:num_training_samples_h4 * signal_length_h4]
#Training_ecg_h4 = np.reshape(Dataset_h4, (-1, signal_length_h4))

import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
h2o.init()

train_ecg_h1 = h2o.H2OFrame(Training_ecg_h1) #train on data with anomalies removed
#train_ecg_h2 = h2o.H2OFrame(Training_ecg_h2) #train on data with anomalies removed
#train_ecg_h3 = h2o.H2OFrame(Training_ecg_h3) #train on data with anomalies removed
#train_ecg_h4 = h2o.H2OFrame(Training_ecg_h4) #train on data with anomalies removed

model_1AE = H2OAutoEncoderEstimator( 
        activation="Tanh", 
        hidden=[100], 
        l1=1e-4,
        score_interval=0,
        epochs=100
)

#model_4 = model_3 = model_2 = model_1

model_1AE.train(x=train_ecg_h1.names, training_frame=train_ecg_h1)
#model_2AE.train(x=train_ecg_h2.names, training_frame=train_ecg_h2)
#model_3AE.train(x=train_ecg_h3.names, training_frame=train_ecg_h3)
#model_4AE.train(x=train_ecg_h4.names, training_frame=train_ecg_h4)

threshold = 0.05
window = 3
Anomalies_1 = np.zeros(signal_length*window)
Anomalies_1[:] = np.nan
Non_anomalies_1 = Anomalies_1
buffers = np.zeros(signal_length)
buffers[:] = np.nan

#Anomalies_4 = Anomalies_3 = Anomalies_2 = Anomalies_1
#Non_anomalies_4 = Non_anomalies_3 = Non_anomalies_2 = Non_anomalies_1

buffers = np.zeros(signal_length)
buffers[:] = np.nan

MSE_anom_1 = np.zeros([len(dataset)/signal_length,2])
MSE_anom_1[:,:] = np.nan
#MSE_non_anom_4 = MSE_anom_4 = MSE_non_anom_3 = MSE_anom_3 = MSE_non_anom_2 = MSE_anom_2 = MSE_non_anom_1 = MSE_anom_1


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

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

import shutil
def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

remove_folder('./ALICE_BANDIT/'+str(Dataset)+'/RUN'+str(run))
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
Error_MSE = np.array([])
#iters = len(dataset) / (signal_num*signal_length)

T = len(dataset)/signal_length
count_1 = 0
count_2 = 0
count_3 = 0
gamma = 0.4965
	
weights = np.ones(K)
P = weights / K
Probs = np.zeros([T,K])
Weights = np.zeros([T,K])
Arms = np.zeros([T,K])
Arms[:,:] = np.nan
Time = np.zeros([T,K])
Time[:] = np.nan
np.random.seed(2)
probs = np.random.rand(T)
signal = signal_num*signal_length

for j in range(0,1):
	for t in range(T):
		print 'Iteration: %.1d' % (t), 'of ', T, '----------------------------------------'
		context = dataset[t*signal_length:t*signal_length+signal]
		test_size = signal_length
		train_size = signal_length
		train, test = context[:len(context)], context[:len(context)]
		validation_split = 0.5
		#reshape into X=t and Y=t+1
		trainX, trainY = create_dataset(train, look_back)
		testX, testY = create_dataset(test, look_back)

		p = probs[t]

		if p >= 0 and p < P[0]:
			print 'p = ', p, ', ARM 1'
			arm = [1,0,0]
			Arms[t,:] = arm
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
			start_10 = time.time()
			history_10 = model_10.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_10 = time.time()
			time_taken_10 = end_10 - start_10
			
			model_11.compile(loss='mean_squared_error', optimizer='adam')
			start_11 = time.time()
			history_11 = model_11.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_11 = time.time()
			time_taken_11 = end_11 - start_11
			
			model_12.compile(loss='mean_squared_error', optimizer='adam')
			start_12 = time.time()
			history_12 = model_11.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_12 = time.time()
			time_taken_12 = end_12 - start_12
			
			Time[t,:] = [time_taken_10,time_taken_11,time_taken_12]
			
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
			if count_1>1:
				os.remove(OnlineFolder+'weights'+str(count_1-1)+'.h5')
			count_1 = count_1 + 1
			reward1 = history_1.history['val_loss']
			reward1 = 1 - reward1[0]
			reward1_est = reward1/P[0]
			P[0] = (1-gamma**t)*(weights[0] / sum(weights))+(gamma**t)/np.float(K)
			P[1] = (1-gamma**t)*(weights[1] / sum(weights))+(gamma**t)/np.float(K)
			P[2] = (1-gamma**t)*(weights[2] / sum(weights))+(gamma**t)/np.float(K)
			Probs[t,:] = P
			print 'Probability distribution: ', P
			weights[0] = weights[0]*np.exp(gamma*reward1_est)
			if weights[0] > 99999:
				weights[0] = 99999
			Weights[t,:] = weights
			testPredict = model_1.predict(testX)
			
			#buffers = np.zeros(signal_length)
			#buffers[:] = np.nan
			#test_AE = np.reshape(testPredict, (-1,len(testPredict)))
			#test_ecg_h1 = h2o.H2OFrame(test_AE)
			#reconstruction_error_h1 = model_1AE.anomaly(test_ecg_h1)
			#df_1 = reconstruction_error_h1.as_data_frame()
			#Pred_AE = np.reshape(test_AE, (test_AE.shape[1]))
			#Pred_AE = Pred_AE[:signal_length]
			#if df_1.values[0,0] > threshold:
			#	Anomalies_1 = np.append(Anomalies_1, Pred_AE)
			#	Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
			#	print 'Signal: %.0d' % (t), 'ANOMALOUS!', df_1.values[0,0]
			#else:
			#	Non_anomalies_1 = np.append(Non_anomalies_1,Pred_AE)
			#	Anomalies_1 = np.append(Anomalies_1,buffers)
			#	print 'Signal: %.0d' % (t), 'Non-anomalous', df_1.values[0,0]
			#Error_MSE = np.append(Error_MSE,df_1.values[0,0])
			
			History_train = np.append(History_train, history_1.history['loss'])
			History_validate = np.append(History_validate, history_1.history['val_loss'])
			Predictions = np.append(Predictions,testPredict[:signal_length])
			
			if t > window:
				results = running_mean(History_validate,window)
				if np.gradient(results)[-1] > 0.5*np.std(np.gradient(results)) or np.gradient(results)[-1] < -0.5*np.std(np.gradient(results)):
					Anomalies_1 = np.append(Anomalies_1,testPredict[:signal_length])
					Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
					print 'Signal: %.0d' % (t), 'ANOMALOUS!'
				else:
					Non_anomalies_1 = np.append(Non_anomalies_1,testPredict[:signal_length])
					Anomalies_1 = np.append(Anomalies_1,buffers)
					print 'Signal: %.0d' % (t), 'Non-anomalous'
			
			model_1.save(OnlineFolder+'model_1_'+str(count_1)+'.h5')
			if count_1>1:
				os.remove(OnlineFolder+'model_1_'+str(count_1-1)+'.h5')
			del model_1
			#print("Model saved to disk.")
			model_1 = load_model(OnlineFolder+'model_1_'+str(count_1)+'.h5')
		elif p >= P[0] and p < (P[0] + P[1]):
			print 'p = ', p, ', ARM 2'
			arm = [0,1,0]
			Arms[t,:] = arm
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
			start_20 = time.time()
			history_20 = model_20.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_20 = time.time()
			time_taken_20 = end_20 - start_20
			
			model_21.compile(loss='mean_squared_error', optimizer='adam')
			start_21 = time.time()
			history_21 = model_21.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_21 = time.time()
			time_taken_21 = end_21 - start_21
			
			model_22.compile(loss='mean_squared_error', optimizer='adam')
			start_22 = time.time()
			history_22 = model_22.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_22 = time.time()
			time_taken_22 = end_22 - start_22
			
			Time[t,:] = [time_taken_20,time_taken_21,time_taken_22]
			
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
			if count_2>1:
				os.remove(OnlineFolder+'weights'+str(count_2-1)+'.h5')
			count_2 = count_2 + 1
			reward2 = history_2.history['val_loss']
			reward2 = 1 - reward2[0]
			reward2_est = reward2/P[1]
			P[0] = (1-gamma**t)*(weights[0] / sum(weights))+(gamma**t)/np.float(K)
			P[1] = (1-gamma**t)*(weights[1] / sum(weights))+(gamma**t)/np.float(K)
			P[2] = (1-gamma**t)*(weights[2] / sum(weights))+(gamma**t)/np.float(K)
			#if t < 3:
			#    P[:] = 1 / np.float(K)
			Probs[t,:] = P
			print 'Probability distribution: ', P
			weights[1] = weights[1]*np.exp(gamma*reward2_est)
			if weights[1] > 99999:
				weights[1] = 99999
			Weights[t,:] = weights
			testPredict = model_2.predict(testX)
			
			#buffers = np.zeros(signal_length)
			#buffers[:] = np.nan
			#test_AE = np.reshape(testPredict, (-1,len(testPredict)))
			#test_ecg_h1 = h2o.H2OFrame(test_AE)
			#reconstruction_error_h1 = model_1AE.anomaly(test_ecg_h1)
			#df_1 = reconstruction_error_h1.as_data_frame()
			#Pred_AE = np.reshape(test_AE, (test_AE.shape[1]))
			#Pred_AE = Pred_AE[:signal_length]
			#if df_1.values[0,0] > threshold:
			#	Anomalies_1 = np.append(Anomalies_1, Pred_AE)
			#	Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
			#	print 'Signal: %.0d' % (t), 'ANOMALOUS!', df_1.values[0,0]
			#else:
			#	Non_anomalies_1 = np.append(Non_anomalies_1,Pred_AE)
			#	Anomalies_1 = np.append(Anomalies_1,buffers)
			#	print 'Signal: %.0d' % (t), 'Non-anomalous', df_1.values[0,0]
			#Error_MSE = np.append(Error_MSE,df_1.values[0,0])
			
			History_train = np.append(History_train, history_2.history['loss'])
			History_validate = np.append(History_validate, history_2.history['val_loss'])
			Predictions = np.append(Predictions,testPredict[:signal_length])
			
			if t > window:
				results = running_mean(History_validate,window)
				if np.gradient(results)[-1] > 0.5*np.std(np.gradient(results)) or np.gradient(results)[-1] < -0.5*np.std(np.gradient(results)):
					Anomalies_1 = np.append(Anomalies_1,testPredict[:signal_length])
					Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
					print 'Signal: %.0d' % (t), 'ANOMALOUS!'
				else:
					Non_anomalies_1 = np.append(Non_anomalies_1,testPredict[:signal_length])
					Anomalies_1 = np.append(Anomalies_1,buffers)
					print 'Signal: %.0d' % (t), 'Non-anomalous'
			
			model_2.save(OnlineFolder+'model_2_'+str(count_2)+'.h5')
			if count_2>1:
				os.remove(OnlineFolder+'model_2_'+str(count_2-1)+'.h5')
			del model_2
			#print("Model saved to disk.")
			model_2 = load_model(OnlineFolder+'model_2_'+str(count_2)+'.h5')
		elif p >= (P[0] + P[1]) and p < 1:
			print 'p = ', p, ', ARM 3'
			arm = [0,0,1]
			Arms[t,:] = arm
			OnlineFolder = BanditFolder+'/Action_3'+'/'
			if not os.path.exists(OnlineFolder):
				os.makedirs(OnlineFolder)   
			if count_3 == 0:
				model_3.load_weights(BanditFolder+'/'+str(model_3)+str(Dataset)+'.h5', by_name=True)
			else:
				model_3.load_weights(OnlineFolder+'weights'+str(count_3-1)+'.h5', by_name=True)
			
			model_30.compile(loss='mean_squared_error', optimizer='adam')
			start_30 = time.time()
			history_30 = model_30.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_30 = time.time()
			time_taken_30 = end_30 - start_30
			
			model_31.compile(loss='mean_squared_error', optimizer='adam')
			start_31 = time.time()
			history_31 = model_31.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_31 = time.time()
			time_taken_31 = end_31 - start_31
			
			model_32.compile(loss='mean_squared_error', optimizer='adam')
			start_32 = time.time()
			history_32 = model_32.fit(trainX, trainY, validation_split=validation_split, epochs=1, batch_size=1, verbose=1)
			end_32 = time.time()
			time_taken_32 = end_32 - start_32
			
			Time[t,:] = [time_taken_30,time_taken_31,time_taken_32]
			
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
			if count_3>1:
				os.remove(OnlineFolder+'weights'+str(count_3-1)+'.h5')
			count_3 = count_3 + 1
			reward3 = history_3.history['val_loss']
			reward3 = 1 - reward3[0]
			reward3_est = reward3/P[2]
			
			P[0] = (1-gamma**t)*(weights[0] / sum(weights))+(gamma**t)/np.float(K)
			P[1] = (1-gamma**t)*(weights[1] / sum(weights))+(gamma**t)/np.float(K)
			P[2] = (1-gamma**t)*(weights[2] / sum(weights))+(gamma**t)/np.float(K)
			#if t < 3:
			#   P[:] = 1 / np.float(K)
			Probs[t,:] = P
			print 'Probability distribution: ', P
			weights[2] = weights[2]*np.exp(gamma*reward3_est)
			if weights[2] > 99999:
				weights[2] = 99999
			Weights[t,:] = weights
			testPredict = model_3.predict(testX)
			
			#buffers = np.zeros(signal_length)
			#buffers[:] = np.nan
			#test_AE = np.reshape(testPredict, (-1,len(testPredict)))
			#test_ecg_h1 = h2o.H2OFrame(test_AE)
			#reconstruction_error_h1 = model_1AE.anomaly(test_ecg_h1)
			#df_1 = reconstruction_error_h1.as_data_frame()
			#Pred_AE = np.reshape(test_AE, (test_AE.shape[1]))
			#Pred_AE = Pred_AE[:signal_length]
			#if df_1.values[0,0] > threshold:
			#	Anomalies_1 = np.append(Anomalies_1, Pred_AE)
			#	Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
			#	print 'Signal: %.0d' % (t), 'ANOMALOUS!', df_1.values[0,0]
			#else:
			#	Non_anomalies_1 = np.append(Non_anomalies_1,Pred_AE)
			#	Anomalies_1 = np.append(Anomalies_1,buffers)
			#	print 'Signal: %.0d' % (t), 'Non-anomalous', df_1.values[0,0]
			#Error_MSE = np.append(Error_MSE,df_1.values[0,0])
			
			History_train = np.append(History_train, history_3.history['loss'])
			History_validate = np.append(History_validate, history_3.history['val_loss'])
			Predictions = np.append(Predictions,testPredict[:signal_length])
			
			if t > window:
				results = running_mean(History_validate,window)
				if np.gradient(results)[-1] > 0.5*np.std(np.gradient(results)) or np.gradient(results)[-1] < -0.5*np.std(np.gradient(results)):
					Anomalies_1 = np.append(Anomalies_1,testPredict[:signal_length])
					Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
					print 'Signal: %.0d' % (t), 'ANOMALOUS!'
				else:
					Non_anomalies_1 = np.append(Non_anomalies_1,testPredict[:signal_length])
					Anomalies_1 = np.append(Anomalies_1,buffers)
					print 'Signal: %.0d' % (t), 'Non-anomalous'
			
			model_3.save(OnlineFolder+'model_3_'+str(count_3)+'.h5')
			if count_3>1:
				os.remove(OnlineFolder+'model_3_'+str(count_3-1)+'.h5')
			del model_3
			#print("Model saved to disk.")
			model_3 = load_model(OnlineFolder+'model_3_'+str(count_3)+'.h5')
np.savetxt('Anomalies_d1a', Anomalies_1)
np.savetxt('Non_anomalies_d1a', Non_anomalies_1)    
np.savetxt('Predictions_d1a', Predictions)
np.savetxt('dataset_d1a', dataset)
np.savetxt('History_train_d1a', History_train)
np.savetxt('History_validate_d1a', History_validate)
#np.savetxt('Error_d2', Error_MSE)
	#f, ax = plt.subplots(3, 1, figsize=(20,8))
	#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
	#				wspace=None, hspace=0.4)

	#Anomalies = np.zeros(len(Correlation))
	#Anomalies[:] = np.nan
	#x = np.linspace(0, len(Correlation), num=len(Correlation))
	#epsilon = np.zeros(len(Correlation))
	#epsilon[:] = 0.95
	#for i in range(len(Correlation)):
	#	if Correlation[i]<0.95:
	#		Anomalies[i] = Correlation[i]
	
	#plt.figure(figsize=(20,10))
	#buffers = np.zeros(signal_num*signal_length)
	#buffers[:] = np.nan
	#dataset1 = np.append(buffers,dataset)
	#Predictions1 = np.append(buffers,Predictions)
	#plt.plot(Predictions, 'r', label='Predicted')
	#plt.plot(dataset[:len(Predictions)], 'b', label='Actual')
	#plt.ylabel('mV')
	#plt.legend(loc='upper left')
	#plt.title('Output predictions')
	#plt.show()

	#ax[1].plot(Probs, linewidth=3)
	#ax[1].plot(Probs[:,0], linewidth=3, label='Arm 1')
	#ax[1].plot(Probs[:,1], linewidth=3, label='Arm 2')
	#ax[1].plot(Probs[:,2], linewidth=3, label='Arm 3')
	#ax[1].set_title('Arm importance weighting')
	#ax[1].legend(loc="upper right")
	#ax[1].set_ylim([-0.1,1.1])

	#ax[2].plot(Correlation, 'k', linewidth=4)
	#ax[2].plot(epsilon, 'r--', linewidth=2)
	#ax[2].plot(Anomalies, 'ro', markersize=10)
	#ax[2].set_title('Anomaly prediction')

	#ax[3].plot(History_train, 'b')
	#ax[3].plot(History_validate, 'r')
	#ax[3].set_ylabel('MSE')
	#ax[3].set_title('Model optimisation')
	
	#plt.show()
	#plt.savefig(BanditFolder+'/Results'+str(run)+'.png')

	#DLB_13 = np.cumsum(History_validate)

	#np.savetxt('DLB_t_1p'+str(j), DLB_13, delimiter=',')
	#np.savetxt('DLB_time1p'+str(j), Time, delimiter=',')
	#np.savetxt('DLB_arms1p'+str(j), Arms, delimiter=',')
