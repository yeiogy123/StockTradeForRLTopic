# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:53:02 2023

@author: asus
"""
import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM,Bidirectional,GRU
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt

class Stock_Pricer():
   def __init__(self,input_length,input_dim=5,save_model_path = "model\\ai_pricer_model.hdf5",model_name="lstm_model"):
       self.model = self.lstm_model2(input_length,input_dim)#輸入天數資料 一天資料的特徵數
       self.save_model_path = save_model_path 
       self.input_length = input_length
   def lstm_model2(self,input_length, input_dim):
       print(input_length, input_dim)
        ############模型架構
       d=0.3
       model= Sequential()
       model.add(LSTM(256,input_shape=(input_length, input_dim),return_sequences=True))
       model.add(Dropout(d))

       model.add(LSTM(128,input_shape=(input_length, input_dim),return_sequences=False))
       model.add(Dropout(d))
       model.add(Dense(16,activation='linear'))
       model.add(Dropout(d))
    
       model.add(Dense(1,activation='linear'))
   
       model.compile(loss='mse',optimizer='adam')
       ###############
       return model 
   def load_model(self,path):
       self.model = tf.keras.models.load_model(path, compile=False)
       
   #預測或訓練資料整理    
   def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, single_step=False):
       data = []
       labels = []

       start_index = start_index + history_size
       if end_index is None:
           end_index = len(dataset) - target_size

       for i in range(start_index, end_index):
           indices = range(i-history_size, i)
           data.append(dataset[indices])
            
           if single_step:
             labels.append(target[i+target_size])
           else:
             labels.append(target[i:i+target_size])
          
       return np.array(data), np.array(labels)
   def train(self,data_x,data_y,_split,_batchSize=32,_epochs=1000):
        print(self.model.summary())

        scaler=MinMaxScaler(feature_range=(0,1))#正規化輸出資料
        data_y = scaler.fit_transform(data_y)
        
        scaler1=MinMaxScaler(feature_range=(0,1))#正規化輸入資料
        data_x = scaler1.fit_transform(data_x)
        
        data_x,data_y=Stock_Pricer.multivariate_data( data_x ,data_y , 0 ,None, self.input_length , 1 ,single_step=True)

        split= _split#驗證集分割度
        
        x_train,y_train  =data_x[:int(split*len(data_x))] ,data_y[:int(split*len(data_x))]
        x_vaild,y_vaild  =data_x[int(split*len(data_x)):] ,data_y[int(split*len(data_x)):]
        my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=300, monitor = 'val_loss')
        ]
        filepath = self.save_model_path 
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min',save_best_only=True)
    
        call_backlist = [ my_callbacks,checkpoint]
        callbacks=call_backlist
        
        historylstm2 = self.model.fit( x_train, y_train, batch_size=_batchSize,shuffle=False , epochs=_epochs,validation_data=(x_vaild,y_vaild),callbacks=call_backlist)
        
        ####顯示訓練模型loss結果
        self.model.summary()
        
        plt.plot(historylstm2.history['loss'])
        plt.plot(historylstm2.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #####
        
   def predict(self,x,y):
        scaler=MinMaxScaler(feature_range=(0,1))
        close = scaler.fit_transform(y)#正規化輸出數據 讓預測數據可以反轉換
        
        scaler1=MinMaxScaler(feature_range=(0,1))
        data = scaler1.fit_transform(x)#正規畫輸入數據

        data , close =Stock_Pricer.multivariate_data( data , close , 0 ,None,self.input_length , 1 ,single_step=True)
        predict_data = self.model.predict(data)#批次預測
        predict_data = scaler.inverse_transform(predict_data).tolist()#反轉換成股價
        return predict_data 
        