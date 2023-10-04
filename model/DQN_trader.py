# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:53:02 2023

@author: asus
"""
from collections import deque
import tensorflow as tf
import random
import numpy as np
import math

class DQN_trader():
  
  def __init__(self, state_size , predition = False , action_num=3, model_name="DQN_trader"): #Stay, Buy, Sell
    
    self.state_size = state_size
    self.action_num = action_num
    self.memory = deque(maxlen=2000)
    self.inventory = []
    self.model_name = model_name
    self.predition = predition
    self.gamma = 0.95#衰減量
    self.epsilon = 1.0#隨機買賣機率 越大代表隨機機率越大
    self.epsilon_final = 0.01#隨機買賣機率衰減限制
    self.epsilon_decay = 0.995#訓練一次 隨機買賣機率機率降低量
    self.model = self.model_dnn()
    
  def model_dnn(self):
    ############模型架構
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(units=16, activation='relu', input_dim=self.state_size))
    
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    #model.add(tf.keras.layers.Dense(units=256, activation='relu'))

    
    model.add(tf.keras.layers.Dense(units=self.action_num, activation='linear'))
    
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-3))
    ###############
    return model

  def load_model(self,path):
      self.model = tf.keras.models.load_model(path, compile=False)
      self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-3))
  def trade(self, state):
    # 隨機買或是預測
    if random.random() <= self.epsilon and not self.predition:
      return random.randrange(self.action_num)
    
    actions = self.model.predict(state)
    ################################買賣策略 可能會隨著訓練次數不同而改變
    p = actions[0][0]/actions[0][1]-1#不買與買的比例  average 0.068
    s = actions[0][2]/actions[0][1]-1#賣與買的比例  average 0.165
    print("p:",actions[0][0]/actions[0][1]-1)
    print("s:",actions[0][2]/actions[0][1]-1)
    #print(actions[0])
    if s < 0.154 and p > 0.061:#0.158 0.065
        return 1
    elif s > 0.17 or p < 0.063:
        return 2
    else:
        return 0
    ###############################
    return np.argmax(actions[0])
  
  def batch_train(self, batch_size):
    batch = []
    #根據batch_size將歷史資料放入
    for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
      batch.append(self.memory[i])
      
    for state, action, reward, next_state, done in batch:
      reward = reward
      if not done:
        reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])#衰減離預測當天的越遠reward
        
      target = self.model.predict(state)
      target[0][action] = reward
      
      self.model.fit(state, target, epochs=1, verbose=0)#訓練
      
    if self.epsilon > self.epsilon_final:
      self.epsilon *= self.epsilon_decay#隨著訓練次數越長 讓模型自己預測 隨機買賣機率降低
  
  def sigmoid(x):
     return 1 / (1 + math.exp(-x))
  def state_creator(self,data, timestep, window_size):
     #建立狀態    
    starting_id = timestep - window_size + 1
     
    if starting_id >= 0:
      windowed_data = data[starting_id:timestep+1]
    else:
      windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])
      
    state = []
    for i in range(window_size - 1):
      state.append(DQN_trader.sigmoid(windowed_data[i+1] - windowed_data[i]))
      
    return np.array([state])