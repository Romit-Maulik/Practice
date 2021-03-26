import os
import time
import yfinance as yf
import dateutil.relativedelta
from datetime import date
import datetime
import numpy as np
import sys
from stocklist import NasdaqController
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed

class mainObj:
    def getData(self, ticker, num_days=30):
        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        pastDate = currentDate - dateutil.relativedelta.relativedelta(days=num_days)
        sys.stdout = open(os.devnull, "w")
        data = yf.download(ticker, pastDate, currentDate)
        sys.stdout = sys.__stdout__

        return data[["Volume"]]

    def find_anomalies(self, data, cutoff):
        anomalies = []
        data_std = np.std(data['Volume'])
        data_mean = np.mean(data['Volume'])
        anomaly_cut_off = data_std * cutoff
        upper_limit = data_mean + anomaly_cut_off
        indexs = data[data['Volume'] > upper_limit].index.tolist()
        outliers = data[data['Volume'] > upper_limit].Volume.tolist()
        index_clean = [str(x.date()) for x in indexs]
        d = {'Dates': index_clean, 'Volume': outliers}
        return d

    def find_anomalies_two(self, data, cutoff):
        indexs = []
        outliers = []
        data_std = np.std(data['Volume'])
        data_mean = np.mean(data['Volume'])
        anomaly_cut_off = data_std * cutoff
        upper_limit = data_mean + anomaly_cut_off
        data.reset_index(level=0, inplace=True)
        for i in range(len(data)):
            temp = data['Volume'].iloc[i]
            if temp > upper_limit:
                indexs.append(data['Date'].iloc[i])
                outliers.append(temp)
        d = {'Dates': indexs, 'Volume': outliers}
        return d

    def parallel_wrapper(self,x, cutoff, currentDate):
        d = (self.find_anomalies_two(self.getData(x), cutoff))
        if d['Dates']:
            for i in range(len(d['Dates'])):
                if self.days_between(str(currentDate), str(d['Dates'][i])) <= 1:
                    self.customPrint(d, x)

    def customPrint(self, d, tick):

        print("\n\n\n*******  " + tick.upper() + "  *******")
        print("Ticker is: "+tick.upper())

        for i in range(len(d['Dates'])):
            str1 = str(d['Dates'][i])
            str2 = str(d['Volume'][i])
            print(str1 + " - " + str2)

        print("*********************\n\n\n")

    def days_between(self, d1, d2):
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
        return abs((d2 - d1).days)

    def main_func(self, cutoff):
        StocksController = NasdaqController(True)
        list_of_tickers = StocksController.getList()
        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

        start_time = time.time()
        Parallel(n_jobs=16)(delayed(self.parallel_wrapper)(x, cutoff, currentDate) for x in tqdm(list_of_tickers) )

        print("\n\n\n\n--- this took %s seconds to run ---" %
              (time.time() - start_time))

if __name__ == '__main__':
    # input desired anomaly standard deviation cuttoff
    # run time around 50 minutes for every single ticker.
    # mainObj().save_data()
    mainObj().main_func(8)
