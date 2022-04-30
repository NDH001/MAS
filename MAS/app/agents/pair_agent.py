from threading import Thread, Semaphore, Event
import requests
import json
import time
from datetime import datetime, tzinfo
import pytz
import warnings

import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()

"""
This class checks correlation between BTC and ETH
and recommends whether to buy or sell BTC if there is a good pattern
"""
class PairAgent:
    def __init__(self):
        self.btc_df = None
        self.eth_df = None
        self.current_date = None
        self.check_new_day()

        self.exit_flag = False
        self.sleep_time = 3600 # 1hr
        self.thread = Thread(name=self.__str__(), target=self.run)

    def start_threads(self):
        # start thread
        self.thread.start()

    def run(self):
        while True:
            self.tick()
            event_obj.wait(self.sleep_time)
            if self.exit_flag:
                break

    def tick(self):
        sem.acquire()
        self.check_new_day()
        print("PairAgent - AdvicePair:{}".format(self.getAdvicePair()))
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('PairAgent - THREAD TERMINATED!')

    def check_new_day(self):
        # Get current datetime in UTC
        utc_now_dt = datetime.now(tz=pytz.UTC).strftime("%m/%d/%Y")

        if self.current_date is None or utc_now_dt != self.current_date:
            self.current_date = utc_now_dt
            print("PairAgent - Getting correlation for:", utc_now_dt)
            self.get_correlation()

    # a function that normalize the btc and eth market price history, so that we can compare them on the same scale.
    # btc and eth shares a correlation of 0.78, although not perfect, but is satisfactory, and the main reason \n
    # for a could be higher correlation is their difference in the earlier years, which in the recent years, they have grown similar to each other, trend wise.
    def get_correlation(self, show = False):
        self.btc_df = pd.DataFrame(yf.Ticker('BTC-USD').history(period='365d',interval ='1d'))
        self.eth_df = pd.DataFrame(yf.Ticker('ETH-USD').history(period='365d',interval ='1d'))

        print(self.btc_df.Close.corr(self.eth_df.Close))
        sc = MinMaxScaler(feature_range = (0, 1))
        self.eth_df["Norm_Close"] = sc.fit_transform(np.asarray(self.eth_df.Close).reshape(-1,1))
        self.btc_df['Norm_Close'] = sc.fit_transform(np.asarray(self.btc_df.Close).reshape(-1,1))

        if show:
            plt.plot(self.eth_df.Norm_Close)
            plt.plot(self.btc_df.Norm_Close)

        self.btc_df['Trade'] = np.zeros(len(self.btc_df))
        self.eth_df['Trade'] = np.zeros(len(self.eth_df))

        #0 do nth
        #2 buy
        #3 sell

        # this part allocates a new column with numbers to the dataset, mainly comparing the correlation between the \n
        # two crypto currencies every timestamp, and if their correlation is less than 0.7 and btc is underperforming, we perform a \n
        # buy action, and if eth is underperforming, we sell. If their correlation is above 0.7, we hold.
        for i in range(2, len(self.eth_df)):
            if self.eth_df.Norm_Close[0:i].corr(self.btc_df.Norm_Close[0:i]) < 0.7:
                # print(self.btc_df.Norm_Close.iloc[i],self.eth_df.Norm_Close.iloc[i])
                if self.btc_df.Norm_Close.iloc[i] < self.eth_df.Norm_Close.iloc[i]:
                    self.btc_df.Trade.iloc[i] = 2
                    self.eth_df.Trade.iloc[i] = 3
                else:

                    self.btc_df.Trade.iloc[i] = 3
                    self.eth_df.Trade.iloc[i] = 2

        if show:
            plt.plot(self.eth_df.Norm_Close,label='ETH')
            plt.plot(self.btc_df.Norm_Close,label='BTC')
            plt.plot(self.btc_df.Norm_Close[self.btc_df.Trade==2],label='Buy chance')
            plt.legend()
            plt.show()


    # call this function for the decision of either buy,sell,or hold. The deciding factor is as above.

    def getAdvicePair(self):
        if self.btc_df is None:
            self.get_correlation()
        if self.btc_df is not None:
            if self.btc_df.Trade[-1] == 2:
                return 'buy'
            elif self.btc_df.Trade[-1] == 3:
                return 'sell'
            else:
                return 'no go'
