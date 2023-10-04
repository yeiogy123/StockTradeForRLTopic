# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:29:53 2023

@author: asus
"""
from tqdm import tqdm_notebook, tqdm
import numpy as np
class RL_Stock_Trainer:
    def __init__(self,model,window_size = 10,episodes = 100,batch_size = 32,save_model_path = "model\\ai_trader_model.h5"):
        self.model = model
        self.window_size = window_size#輸入天數
        self.episodes = episodes#迭帶次數
        self.batch_size = batch_size#每次迭代的訓練資料量
        self.save_model_path = save_model_path#保存模型位置
    def stocks_price_format(n):#輸出股價格式
        if n < 0:
            return "- $ {0:2f}".format(abs(n))
        else:
            return "$ {0:2f}".format(abs(n))
    def logData(BuyTime,SellTime,EarnTime,LoseTime,TotalProfit):#輸出損益檔案 觀察用
        f = open("model\\log\\Trading_Bot_Data.txt","a+")
        f.write( "bt = %d,st = %d,et = %d,lt = %d,tp = %d\n" %(BuyTime,SellTime,EarnTime,LoseTime,TotalProfit))
        f.close()
    def train(self,data):
        trader = self.model
        data_samples = len(data) - 1#資料量
        for episode in range(1, self.episodes + 1):
  
          print("Episode: {}/{}".format(episode, self.episodes))
          
          state = trader.state_creator(data, 0, self.window_size + 1)#建立初始狀態
          total_profit = 0
          trader.inventory = []#庫存初始化
          buy_time = 0
          sell_time = 0
          earn_time = 0
          lose_time = 0
          for t in tqdm(range(data_samples)):
            
            action = trader.trade(state)#判斷該狀態預測動作
            
            next_state = trader.state_creator(data, t+1, self.window_size + 1)#建立下一個狀態
            reward = 0
            
            if action == 1: #買
              trader.inventory.append(data[t])#加入庫存
              print("DQN Trader bought: ",  RL_Stock_Trainer.stocks_price_format(data[t]))
              buy_time = buy_time + 1
              
            elif action == 2 and len(trader.inventory) > 0: #賣
              buy_price = trader.inventory.pop(0)#移出庫存
              sell_time = sell_time + 1
              reward = max(data[t] - buy_price, 0)#計算回饋
              total_profit += data[t] - buy_price
              if (data[t] - buy_price) > 0:
                  earn_time = earn_time + 1
              elif (data[t] - buy_price) < 0:
                  lose_time = lose_time + 1
              print("DQN Trader sold: ",  RL_Stock_Trainer.stocks_price_format(data[t]), " Profit: " +  RL_Stock_Trainer.stocks_price_format(data[t] - buy_price) )
              
            if t == data_samples - 1:#選擇訓練日期區間是否結束
              done = True
            else:
              done = False
              
            trader.memory.append((state, action, reward, next_state, done))#將訓練資料加入佇列
            
            state = next_state
            
            if done:
              RL_Stock_Trainer.logData(buy_time,sell_time,earn_time,lose_time,total_profit)
              print("########################")
              print("TOTAL PROFIT: {}".format(total_profit))
              print("########################")
            
            if len(trader.memory) > self.batch_size:#判斷佇列是否大於訓練資料量
              trader.batch_train(self.batch_size)#訓練
              
          if episode % 3 == 0:#每迭代次數保存模型
            trader.model.save(self.save_model_path)
            trader.memory.clear()