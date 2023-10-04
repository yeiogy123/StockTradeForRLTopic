
import numpy as np
import matplotlib.pyplot as plt
from model.DQN_trader import DQN_trader
from model.RL_Stock_Trainer import RL_Stock_Trainer
from Tool.Dataset_Loader import Dataset_Loader
import os


if __name__ == "__main__":  
    stock_name = "2330"#要訓練得股票ID
    train_data_type = ['close']#需要訓練的輸入特徵
    dl = Dataset_Loader(stock_name,"20220701","20230214")#建立日期間股價資料組
    close = dl.get(train_data_type)#找尋資料組內的特定特徵資料
    close = [i[0] for i in close]#訓練時需要的型態修改
    episodes = 100#迭帶次數
    batch_size = 32#每個iteration以32筆做計算
    window_size = 10#模型輸入資料數
    hist_model_path = "model\\ai_trader_model.h5"
    trader = DQN_trader(window_size,False)#False代表訓練模式
    if os.path.isfile(hist_model_path):
        print("load model from %s" % hist_model_path)
        trader.load_model(hist_model_path)
    trainer = RL_Stock_Trainer(trader,trader.state_size,episodes,batch_size)#RL訓練環境類別創建
    trainer.train(close) #RL訓練