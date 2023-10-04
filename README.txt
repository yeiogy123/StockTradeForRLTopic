===============
Trading_Bot.py

Trading_Bot(RL_Moel,Stock_Price_Model)
透過RL_Model增強式訓練模型與Stock_Price_Model監督式訓練模型判斷買賣與買價

Trading_Bot.run(data,close)
放入訓練時用的輸入資料與輸出資料(正規化反轉會用到)來回傳買賣動作與價錢

===============
Stock_Pricer_TrainOrTest.py

Stock_Pricer(input_length,input_dim=5,save_model_path = "model\\ai_pricer_model.hdf5",model_name="lstm_model")
input_length輸入模型天數 ，input_dim各天數股票指標數值特徵

Stock_Pricer.train(data_x,data_y,_split,_batchSize=32,_epochs=1000)
data_x輸入天數期間各項數值指標，data_y輸出結果，_split分割訓練集做驗證來訓練

Stock_Pricer.predict(x,y)
x預測天數期間各項數值指標，y輸出結果(正規化反轉會用到)，正規化反轉得到真正的股價

===============
RL_Trader_Train.py

DQN_trader(state_size , predition = False , action_num=3, model_name="DQN_trader")
state_size 輸入模型天數，predition是否為預測模式，action_num買賣與不做透過數值表示，建立模型

DQN_trader.batch_train(batch_size)
batch_size每次迭代訓練送進去模型的資料量來進行訓練

DQN_trader.state_creator(data, timestep, window_size)
data所有訓練資料集，timestep資料集起始位置，window_size起始位置向後取多少資料，取出的資料建立狀態

RL_Stock_Trainer(model,window_size = 10,episodes = 100,batch_size = 32,save_model_path = "model\\ai_trader_model.h5")
model需要用來增強式學習的模型，window_size 輸入天數，episodes迭代次數，來建立增強式學習環境

RL_Stock_Trainer.train(data)
data訓練天數期間資料來訓練
================