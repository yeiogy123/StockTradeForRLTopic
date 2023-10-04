# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:36:39 2023

@author: asus
"""

import datetime
from keras.models import load_model
import numpy as np
from model.Stock_Pricer import Stock_Pricer
from model.DQN_trader import DQN_trader
from Tool.Stock_Tool import Stock_Tool
from model.Stock_Pricer import Stock_Pricer
from Tool.Dataset_Loader import Dataset_Loader
import matplotlib.pyplot as plt
import os
import tensorflow


class Trading_Bot():
     def __init__(self,RL_Moel,Stock_Price_Model):
         self.agent = RL_Moel
         self.Stock_Pricer = Stock_Price_Model
     def stock_price_ex(x):
         x = round(x,1)
         y = int(x*10)%10
         if y>=5:
             return int(x)+0.5
         else:
             return int(x)
     def run(self,data,close):
         print(len(close))
         trade_close = [i[0] for i in close]
         state = self.agent.state_creator( trade_close , 0, window_size + 1)#計算初始狀態
         l = len(close)-1
         predict_price = self.Stock_Pricer.predict(data,close)#預測價錢
         for t in range(l):
            action = self.agent.trade(state)#迭帶至最後一個action
            next_state = self.agent.state_creator( trade_close , t + 1, window_size + 1)#建立日期區間狀態
            done = True if t == l - 1 else False
            state = next_state
            print('action=',action,'price=',Trading_Bot.stock_price_ex(predict_price[-1][0]))
            if done:
                return action , Trading_Bot.stock_price_ex(predict_price[-1][0])
         return None
                

if __name__ == "__main__":
    one = True
    #########建立資料
    while(True and one ):
        print('input stock id=')
        Stock_ID = input()
        #Stock_ID = "0056"#要購買股票ID
        now = datetime.datetime.now()#預測今日價格需要到今日
        today_date = now.strftime("%Y%m%d")#根據日期排列
        input_data_type = ['open','high','low','transaction_volume','close']#預測價格使用到輸入的股價特徵
        output_data_type = ['close']#預測價格使用到輸出的股價特徵
        dl = Dataset_Loader(Stock_ID,"20221210",'20230608')#建立股價資料組 需要建立RL狀態間隔時間 建議設置一個月以上
        data = dl.get(input_data_type)#從資料組中找尋特定資料
        print('data=', data)
        close = dl.get(output_data_type)
        print('close=', close)
    ##########
    
    ########RL股價買賣機器人建立
        model =  tensorflow.keras.models.load_model("model\\ai_trader_model.h5", compile=False)
        window_size = model.layers[0].input.shape.as_list()[1]
        RL_bot = DQN_trader(window_size,True)#創建RL機器人 True代表預測模式
        if os.path.isfile("model\\ai_pricer_model.hdf5"):
             RL_bot.load_model("model\\ai_trader_model.h5")
        #print(RL_bot.model.summary())
        #RL_bot.batch_train(2)
    ##########

    #########預測股價模型
        stock_pricer =  Stock_Pricer(window_size)
        #if os.path.isfile("model\\ai_pricer_model.hdf5"):
         #   print("load model from %s" % "model\\ai_pricer_model.hdf5")
          #  stock_pricer.load_model("model\\ai_pricer_model.hdf5")
        stock_pricer.train(data,close,0.995,64,1000)
    #########
    
    #########預測買賣與股價模型
        print('len=',len(close))
        agent = Trading_Bot(RL_bot,stock_pricer)#將RL股價買賣機器人 與 股價預測模型結合
        act,predict_price = agent.run(data,close)#預測動作及價格

    #########
    ########會員登入
        user = Stock_Tool("f74118227","18227")#會員登入
        user_inventory = user.Get_User_Stocks()#查看庫存
        if act==1:
            user.Buy_Stock(Stock_ID,1,predict_price)
        elif act==2 and len(user_inventory)>0:
            user.Sell_Stock(Stock_ID,1, predict_price)
        if act==1 or act==2 :
            break
        print("agent predict => action:%d , price:%f" % (act,predict_price))#0不動作  1買入  2賣出
    ########