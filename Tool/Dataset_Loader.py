# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:09:23 2023

@author: asus
"""
from .Stock_Tool import Stock_Tool

###股價資料組
class Dataset_Loader:
    def __init__(self,stock_name,start_date,end_date):
        self.dataset = Stock_Tool.Get_Stock_Informations(stock_name, start_date, end_date )#建立股價資料(股票ID,起始時間,結束時間)
    def get(self,data_type):#找尋特定特徵資料
        data = []
        for i in self.dataset:
            tmp = []
            for j in data_type:
                tmp.append(i[j])
            data.append(tmp)
        data = data[::-1]
        return data
    def change(self,stock_name,start_date,end_date):#改變查詢資料(股票ID,起始時間,結束時間)
        self.dataset = Stock_Tool.Get_Stock_Informations(stock_name, start_date, end_date )