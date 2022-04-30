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

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()

"""
Bollinger Band Agent has 3 bands, Low, Med, High
Low: SMA - 2*s.d.
Med: SMA
High: SMA + 2*s.d.
If price > High, 'overbuy'
If price < Low, 'oversell'
"""

class BollingerBandAgent:
    def __init__(self):
        self.current_date = None
        self.btc_df = None
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
        # update SMA
        self.check_new_day()
        print("BollingerBandAgent - BB:{}".format(self.get_Advice_BB()))
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('BollingerBandAgent - THREAD TERMINATED!')

    def check_new_day(self):
        # Get current datetime in UTC
        utc_now_dt = datetime.now(tz=pytz.UTC).strftime("%m/%d/%Y")

        if self.current_date is None or utc_now_dt != self.current_date:
            self.current_date = utc_now_dt
            print("BollingerBandAgent - Getting BB for:", utc_now_dt)
            self.get_bollinger_bands()

    #function to get the historical data of btc, dropped dividien, stock splits for tidiness
    def get_bollinger_bands(self):
        df = pd.DataFrame(yf.Ticker('BTC-USD').history(period='365d',interval ='1d'))
        self.btc_df = df.drop(['Dividends','Stock Splits'],1)
        self.btc_BB()

    #calcuates the bollinger band using the sma and related math. Set up overbuy or oversell signals for visual representation \n
    # and suggestion making
    def btc_BB(self, show=False):
        # calculate simple moving average and standard deviation
        sma = self.btc_df.Close.rolling(window=20, min_periods=1).mean()
        sma_sd = self.btc_df.Close.rolling(window=20).std()

        # calculate low and high bands
        self.btc_df['Low_band_20_SMA'] = sma - (sma_sd * 2)
        self.btc_df['Med_band_20_SMA'] = sma
        self.btc_df['High_band_20_SMA'] = sma + (sma_sd * 2)

        # calculate signals
        self.btc_df['Overbuy'] = np.where(self.btc_df['Close'] >= self.btc_df['High_band_20_SMA'], 1, 0)
        self.btc_df['Overbuy_diff'] = self.btc_df['Close'] - self.btc_df['High_band_20_SMA']
        self.btc_df['Sell_signal'] = self.btc_df['Overbuy'].diff()

        self.btc_df['Oversell'] = np.where(self.btc_df['Close'] <= self.btc_df['Low_band_20_SMA'], 1, 0)
        self.btc_df['Oversell_diff'] = self.btc_df['Close'] - self.btc_df['Low_band_20_SMA']
        self.btc_df['Buy_signal'] = self.btc_df['Oversell'].diff()

        if show:
            plt.figure(figsize = (25,13))

            plt.plot(self.btc_df.index,self.btc_df['Close'],color='k')
            self.btc_df['Low_band_20_SMA'].plot(color = 'orange',label = 'Low Band')
            self.btc_df['Med_band_20_SMA'].plot(color = 'black',label = 'Medium Band',alpha=0.3)
            self.btc_df['High_band_20_SMA'].plot(color='orange', label='High Band', alpha=0.3)

            plt.plot(self.btc_df[self.btc_df['Oversell'] == 1].index,
                     self.btc_df['Low_band_20_SMA'][self.btc_df['Oversell'] == 1],
                     '^', markersize = 15, color = 'g', label = 'Oversell')

            plt.plot(self.btc_df[self.btc_df['Overbuy'] == 1].index,
                     self.btc_df['High_band_20_SMA'][self.btc_df['Overbuy'] == 1],
                     'v', markersize = 15, color = 'r', label = 'Overbuy')

            plt.ylabel('Closing price', fontsize = 15 )
            plt.xlabel('Date', fontsize = 15 )
            plt.title('Bollinger Bands', fontsize = 20)
            plt.legend()
            plt.grid()
            plt.show()

    # function to get advice on bollinger band based on the signal, if overbuy signal, we buy, if oversell, we sell

    def get_Advice_BB(self):
        overall_signal = 0
        if self.btc_df.Buy_signal[-1] == 1:
            overall_signal = 1

        if self.btc_df.Sell_signal[-1] == 1:
            if overall_signal == 1:
                # find which signal is stronger
                signal_diff = self.btc_df.Overbuy_diff[-1] - self.btc_df.Oversell_diff[-1]
                if signal_diff >= 0:
                    overall_signal = 1
                else:
                    overall_signal = -1

        if overall_signal == 1:
            return 'buy'
        elif overall_signal == -1:
            return 'sell'
        else:
            return 'no go'

if __name__ == "__main__":
    bba = BollingerBandAgent()