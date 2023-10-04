import numpy as np
from Tool.Stock_Tool import Stock_Tool
import matplotlib.pyplot as plt
from model.Stock_Pricer import Stock_Pricer
from Tool.Dataset_Loader import Dataset_Loader
import os

def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

###輸出成股票的正式價錢    
def stock_price_ex(x):
    x = round(x,1)
    y = int(x*10)%10
    if y>=5:
        return int(x)+0.5
    else:
        return int(x)    

if __name__ == "__main__":  
    prediction = True#True預測  False訓練
    stock_name = "2330"
    input_data_type = ['open','high','low','transaction_volume','close']#輸入個別天數的特徵資料
    output_data_type = ['close']#輸出特徵資料
    dl = Dataset_Loader(stock_name,"20220101","20230306")#建立資料組
    data = dl.get(input_data_type)#找尋特定輸入特徵資料組
    close = dl.get(output_data_type)#找尋特定輸出特徵資料組
    plt.plot(close)
    plt.show()
    
    
    hist_model_path = "model\\ai_pricer_model.hdf5"
    window_size = 10#輸入天數
    stock_pricer =  Stock_Pricer(window_size)
    if os.path.isfile(hist_model_path):
        print("load model from %s" % hist_model_path)
        stock_pricer.load_model(hist_model_path)
    if not  prediction:
        stock_pricer.train(data,close,0.95,32,1000)#訓練資料 輸出資料 分割度 每次迭帶資料數 迭帶次數
    else:
        predict_data = stock_pricer.predict(data,close)#預測資料 輸出資料(正規化反轉數據用)
        predict_price = [[stock_price_ex(i[0])] for i in predict_data]
        plt.plot(close[(len(close)-len(predict_price)):len(close)])
        plt.plot(predict_price)
        plt.ylabel('price')
        plt.xlabel('step')
        plt.legend(['Realy','Train'])
        plt.show()
