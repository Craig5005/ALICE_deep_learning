import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import animation, rc
from IPython.display import HTML

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

signal_num = 2

signal_length_h1 = 160
signal_length_h2 = 190
signal_length_h3 = 130
signal_length_h4 = 170

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

num_training_samples_h2 = int(len(healthy_person_2) / signal_length_h2)
Dataset_h2 = healthy_person_2[:num_training_samples_h2 * signal_length_h2]
Training_ecg_h2 = np.reshape(Dataset_h2, (-1, signal_length_h2))

num_training_samples_h3 = int(len(healthy_person_3) / signal_length_h3)
Dataset_h3 = healthy_person_3[:num_training_samples_h3 * signal_length_h3]
Training_ecg_h3 = np.reshape(Dataset_h3, (-1, signal_length_h3))

num_training_samples_h4 = int(len(healthy_person_4) / signal_length_h4)
Dataset_h4 = healthy_person_4[:num_training_samples_h4 * signal_length_h4]
Training_ecg_h4 = np.reshape(Dataset_h4, (-1, signal_length_h4))

import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
h2o.init()

train_ecg_h1 = h2o.H2OFrame(Training_ecg_h1) #train on data with anomalies removed
train_ecg_h2 = h2o.H2OFrame(Training_ecg_h2) #train on data with anomalies removed
train_ecg_h3 = h2o.H2OFrame(Training_ecg_h3) #train on data with anomalies removed
train_ecg_h4 = h2o.H2OFrame(Training_ecg_h4) #train on data with anomalies removed

model_1 = H2OAutoEncoderEstimator( 
        activation="Tanh", 
        hidden=[100], 
        l1=1e-4,
        score_interval=0,
        epochs=100
)

model_4 = model_3 = model_2 = model_1

model_1.train(x=train_ecg_h1.names, training_frame=train_ecg_h1)
model_2.train(x=train_ecg_h2.names, training_frame=train_ecg_h2)
model_3.train(x=train_ecg_h3.names, training_frame=train_ecg_h3)
model_4.train(x=train_ecg_h4.names, training_frame=train_ecg_h4)

threshold = 0.015

Anomalies_1 = []
Non_anomalies_1 = []

Anomalies_4 = Anomalies_3 = Anomalies_2 = Anomalies_1
Non_anomalies_4 = Non_anomalies_3 = Non_anomalies_2 = Non_anomalies_1

buffers = np.zeros(signal_length)
buffers[:] = np.nan

MSE_anom_1 = np.zeros([len(dataset)/signal_length,2])
MSE_anom_1[:,:] = np.nan
MSE_non_anom_4 = MSE_anom_4 = MSE_non_anom_3 = MSE_anom_3 = MSE_non_anom_2 = MSE_anom_2 = MSE_non_anom_1 = MSE_anom_1

for i in range(len(dataset)/signal_length):
    Predicted_ecg = dataset[i*signal_length:(i+1)*signal_length]
    Predicted_ecg = np.reshape(Predicted_ecg, (-1,signal_length))
    
    test_ecg_h1 = h2o.H2OFrame(Predicted_ecg) #test using predicted time series
    reconstruction_error_h1 = model_1.anomaly(test_ecg_h1)
    df_1 = reconstruction_error_h1.as_data_frame()
    
    test_ecg_h2 = h2o.H2OFrame(Predicted_ecg) #test using predicted time series
    reconstruction_error_h2 = model_2.anomaly(test_ecg_h2)
    df_2 = reconstruction_error_h2.as_data_frame()
    
    test_ecg_h3 = h2o.H2OFrame(Predicted_ecg) #test using predicted time series
    reconstruction_error_h3 = model_3.anomaly(test_ecg_h3)
    df_3 = reconstruction_error_h3.as_data_frame()
    
    test_ecg_h4 = h2o.H2OFrame(Predicted_ecg) #test using predicted time series
    reconstruction_error_h4 = model_4.anomaly(test_ecg_h4)
    df_4 = reconstruction_error_h4.as_data_frame()
    
    
    #anomalies = df_sorted[ df_sorted['Reconstruction.MSE'] > threshold ]
    #not_anomalies = df_sorted[ df_sorted['Reconstruction.MSE'] < threshold ]
    Predictions = np.reshape(Predicted_ecg, (Predicted_ecg.shape[1]))
    if df_1.values[0,0] > threshold:
        Anomalies_1 = np.append(Anomalies_1,Predictions)
        Non_anomalies_1 = np.append(Non_anomalies_1,buffers)
        MSE_anom_1[i,:] = [np.int(i),df_1.values[0,0]]
        print MSE_anom_1[i,:]
        print 'Anomalous 1'
    else:
        Non_anomalies_1 = np.append(Non_anomalies_1,Predictions)
        Anomalies_1 = np.append(Anomalies_1,buffers)
        MSE_non_anom_1[i,:] = [np.int(i),df_1.values[0,0]]
        print MSE_non_anom_1[i,:]
        print 'Non-anomalous 1'
    if df_2.values[0,0] > threshold:
        Anomalies_2 = np.append(Anomalies_2,Predictions)
        Non_anomalies_2 = np.append(Non_anomalies_2,buffers)
        MSE_anom_2[i,:] = [np.int(i),df_2.values[0,0]]
        print MSE_anom_2[i,:]
        print 'Anomalous 2'
    else:
        Non_anomalies_2 = np.append(Non_anomalies_2,Predictions)
        Anomalies_2 = np.append(Anomalies_2,buffers)
        MSE_non_anom_2[i,:] = [np.int(i),df_2.values[0,0]]
        print MSE_non_anom_2[i,:]
        print 'Non-anomalous 2'
    if df_3.values[0,0] > threshold:
        Anomalies_3 = np.append(Anomalies_3,Predictions)
        Non_anomalies_3 = np.append(Non_anomalies_3,buffers)
        MSE_anom_3[i,:] = [np.int(i),df_3.values[0,0]]
        print MSE_anom_3[i,:]
        print 'Anomalous 3'
    else:
        Non_anomalies_3 = np.append(Non_anomalies_3,Predictions)
        Anomalies_3 = np.append(Anomalies_3,buffers)
        MSE_non_anom_3[i,:] = [np.int(i),df_3.values[0,0]]
        print MSE_non_anom_3[i,:]
        print 'Non-anomalous 3'
    if df_4.values[0,0] > threshold:
        Anomalies_4 = np.append(Anomalies_4,Predictions)
        Non_anomalies_4 = np.append(Non_anomalies_4,buffers)
        MSE_anom_4[i,:] = [np.int(i),df_4.values[0,0]]
        print MSE_anom_4[i,:]
        print 'Anomalous 4'
    else:
        Non_anomalies_4 = np.append(Non_anomalies_4,Predictions)
        Anomalies_4 = np.append(Anomalies_4,buffers)
        MSE_non_anom_4[i,:] = [np.int(i),df_4.values[0,0]]
        print MSE_non_anom_4[i,:]
        print 'Non-anomalous 4'
        
f, ax = plt.subplots(2, 2, figsize=(20,7))

ax[0,0].plot(Anomalies_1, 'r')
ax[0,0].plot(Non_anomalies_1, 'b')
ax[0,0].set_title('AE h1')

ax[0,1].plot(Anomalies_2, 'r')
ax[0,1].plot(Non_anomalies_2, 'b')
ax[0,1].set_title('AE h2')

ax[1,0].plot(Anomalies_3, 'r')
ax[1,0].plot(Non_anomalies_3, 'b')
ax[1,0].set_title('AE h3')

ax[1,1].plot(Anomalies_4, 'r')
ax[1,1].plot(Non_anomalies_4, 'b')
ax[1,1].set_title('AE h4')

plt.show()
