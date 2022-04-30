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
This is the Moving Average Agent
It will get SMA and EMA daily
"""


class MovingAverageAgent:
    def __init__(self):
        self.current_date = None
        self.btc_df = None
        self.check_new_day()

        self.exit_flag = False
        self.sleep_time = 3600 # 1hr
        self.thread = Thread(name=self.__str__(), target=self.run)

    #function to start the flask container
    def start_threads(self):
        # start thread
        self.thread.start()

    #loops the update of flask
    def run(self):
        while True:
            self.tick()
            event_obj.wait(self.sleep_time)
            if self.exit_flag:
                break
    #loops every predetermined timestamp
    def tick(self):
        sem.acquire()
        # update EMA and SMA
        self.check_new_day()
        print("MovingAverageAgent - SMA:{}, EMA{}".format(self.getAdviceSMA(), self.getAdviceEMA()))
        sem.release()
    #stops flask
    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('MovingAverageAgent  - THREAD TERMINATED!')

    def check_new_day(self):
        # Get current datetime in UTC
        utc_now_dt = datetime.now(tz=pytz.UTC).strftime("%m/%d/%Y")

        if self.current_date is None or utc_now_dt != self.current_date:
            self.current_date = utc_now_dt
            print("MovingAverageAgent - Getting moving averages for:", utc_now_dt)
            self.get_moving_averages()


# function to get both the simple and exponential moving average by first retrieving data from yfinance in a period of 365 days, with a interval of 1 day
# for simplicity, we want to drop dividends and stock splits since we are not going to use them in anyways
    def get_moving_averages(self):
        df = pd.DataFrame(yf.Ticker('BTC-USD').history(period='365d',interval ='1d'))
        self.btc_df = df.drop(['Dividends','Stock Splits'],1)
        self.btc_SMA()
        self.btc_EMA()

# function to get the historical price of BTC by selecting the index and close column
    def btc_history(self):
        plt.plot(self.btc_df.index,self.btc_df.Close)
        plt.ylabel('Price', fontsize = 10 )
        plt.xlabel('Date', fontsize = 10 )
        plt.title('BTC-USD', fontsize = 15)

# function to get the simple moving average using the rolling function from the pandas package.
# also added in a show variable for debugging and visualization if there is anything goes wrong in the future.
# this function would indicate the buy and sell points by calculating the 20 and 50 days sma and their cross section
    def btc_SMA(self, show=False):
        self.btc_df['20_SMA'] = self.btc_df.Close.rolling(window = 20, min_periods = 1).mean()
        self.btc_df['50_SMA'] = self.btc_df.Close.rolling(window = 50, min_periods = 1).mean()
        self.btc_df.head()

        self.btc_df['Signal'] = 0.0
        self.btc_df['Signal'] = np.where(self.btc_df['20_SMA'] >= self.btc_df['50_SMA'], 1.0, -1.0)
        self.btc_df['Position'] = self.btc_df['Signal'].diff()

        if show:
            plt.figure(figsize = (20,10))

            plt.plot(self.btc_df.index,self.btc_df['Close'],color='k')
            self.btc_df['20_SMA'].plot(color = 'b',label = '20-day SMA')
            self.btc_df['50_SMA'].plot(color = 'orange',label = '50-day SMA')

            plt.plot(self.btc_df[self.btc_df['Position'] == 1].index,
                     self.btc_df['20_SMA'][self.btc_df['Position'] == 1],
                     '^', markersize = 15, color = 'g', label = 'buy')

            plt.plot(self.btc_df[self.btc_df['Position'] == -1].index,
                     self.btc_df['20_SMA'][self.btc_df['Position'] == -1],
                     'v', markersize = 15, color = 'r', label = 'sell')

            plt.ylabel('Closing price', fontsize = 15 )
            plt.xlabel('Date', fontsize = 15 )
            plt.title('EOSUSD', fontsize = 20)
            plt.show()


#function that does similar task as to the function above, but instead of sma, it is ema this time.
    def btc_EMA(self, show=False):
        self.btc_df['20_EMA'] = self.btc_df['Close'].ewm(span = 20, adjust = False).mean()
        self.btc_df['50_EMA'] = self.btc_df['Close'].ewm(span = 50, adjust = False).mean()
        self.btc_df['Signal_EMA'] = 0.0
        self.btc_df['Signal_EMA'] = np.where(self.btc_df['20_EMA'] > self.btc_df['50_EMA'], 1.0, 0.0)
        self.btc_df['Position_EMA'] = self.btc_df['Signal_EMA'].diff()

        if show:
            plt.figure(figsize = (25,13))

            plt.plot(self.btc_df.index,self.btc_df['Close'],color='k')
            self.btc_df['20_EMA'].plot(color = 'b',label = '20-day SMA')
            self.btc_df['50_EMA'].plot(color = 'b',label = '50-day SMA',alpha=0.3)

            plt.plot(self.btc_df[self.btc_df['Position_EMA'] == 1].index,
                     self.btc_df['20_EMA'][self.btc_df['Position_EMA'] == 1],
                     '^', markersize = 15, color = 'g', label = 'EMA buy')

            plt.plot(self.btc_df[self.btc_df['Position_EMA'] == -1].index,
                     self.btc_df['20_EMA'][self.btc_df['Position_EMA'] == -1],
                     'v', markersize = 15, color = 'r', label = 'EMA sell')

            plt.ylabel('Closing price', fontsize = 15 )
            plt.xlabel('Date', fontsize = 15 )
            plt.title('EOSUSD', fontsize = 20)
            plt.legend()
            plt.grid()
            plt.show()

#a function that returns either buy or sell based on the signal of the last row of the sma calculations
    def getAdviceSMA(self):
        if self.btc_df.Signal[-1] == 1:
            return 'buy'
        elif self.btc_df.Signal[-1] == -1:
            return 'sell'

# a function that returns either buy or sell based on the signal of the last row of the ema calculations
    def getAdviceEMA(self):
        if self.btc_df.Signal_EMA[-1] == 1:
            return 'buy'
        elif self.btc_df.Signal_EMA[-1] == -1:
            return 'sell'