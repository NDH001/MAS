from threading import Thread, Semaphore, Event
import requests
import json
import time
import numpy as np
from datetime import datetime, tzinfo
import pytz
import warnings

import ccxt
import pandas as pd

import random
import re
import random, string
from random import randrange

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()

api_key = "EaYD7jOSDJTFp4s_RXHMv6wrd_e3msPe"
secret = "sA-PYfrgIj5PVbTdiZ_2g9ScqGVYEJie"

"""
This is the broker
It will make transactions based on the final decision from CEO
"""
class BrokerAgent:
    def __init__(self, mode, balance, sql_logger):
        # record transactions
        self.sql_logger = sql_logger

        # advice_only  OR simulated
        self.mode = mode

        # simulated balances
        self.advice_balance_usd = float(balance)
        self.advice_balance_btc = 0.0
        self.advice_balance_eth = 0.0

        # use ccxt encapsulate hitbtc api
        # https://docs.ccxt.com/en/latest/manual.html
        self.hitbtc = ccxt.hitbtc({
            'apiKey': api_key,
            'secret': secret,
            'urls': {
                'api': {
                    'private': 'https://api.demo.hitbtc.com'
                }
            },
            'verbose': False
        })

        self.pnl_agent = None
        self.ceo_agent = None

        self.exit_flag = False
        self.sleep_time = 3600 # 1hr
        self.thread = Thread(name=self.__str__(), target=self.run)

    def init_agents(self, pnl_agent, ceo_agent):
        # set agents
        self.pnl_agent = pnl_agent
        self.ceo_agent = ceo_agent
        # any req methods

    def start_threads(self):
        # start thread
        self.thread.start()

    def run(self):
        sem.acquire()
        while True:
            self.tick()
            event_obj.wait(self.sleep_time)
            if self.exit_flag:
                break
        sem.release()

    def tick(self):
        sem.acquire()
        self.make_trade()
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('BrokerAgent - THREAD TERMINATED!')

    # get latest data
    def fetch_data(self):
        if self.mode == "Simulated":
            usd_balance, btc_balance, eth_balance = 0.0, 0.0, 0.0
            usd_balance = float(self.get_balance("USD"))
            btc_balance = float(self.get_balance("BTC"))
            eth_balance = float(self.get_balance("ETH"))
        else:
            usd_balance = self.advice_balance_usd
            btc_balance = self.advice_balance_btc
            eth_balance = self.advice_balance_eth

        btc_price = float(self.get_ticker_price("BTCUSD"))
        eth_price = float(self.get_ticker_price("ETHUSD"))
        return usd_balance, btc_balance, eth_balance, btc_price, eth_price


    # make trade
    def make_trade(self):
        print("BrokerAgent - Making a trade")
        if self.pnl_agent is not None and self.ceo_agent is not None:

            # get full info from PNLAgent
            prev_state = self.pnl_agent.get_prev_state()
            # print('pnl prev state:', prev_state)

            # get CEO's decision
            action, symbol, amount = self.ceo_agent.get_final_decision()
            print("broker debug", action, symbol, amount)

            if action is not None:
                if self.mode == "Simulated":
                    self.sim_transaction(action, symbol, amount, prev_state)
                elif self.mode == "Advice_Only":
                    self.advice_transaction(action, symbol, amount, prev_state)
        else:
            print("BrokerAgent - other agents not initialized")

    def sim_transaction(self, action, symbol, amount, prev_state, price=None):
        if action == -1 and symbol == "BTC":
            # sell BTC
            btc_balance = prev_state["USD_Balance"]
            if btc_balance >= amount:
                # specify price or sell order
                if price is None:
                    print("BrokerAgent - place sell order at current price")
                    result = self.place_market_sell_order(symbol, amount)
                    self.pnl_agent.update_state()
                    self.sql_logger.process_simulated(result)
                else:
                    print("BrokerAgent - place sell order at specified price")
                    result = self.place_limit_sell_order(symbol, amount, price)
                    self.pnl_agent.update_state()
                    self.sql_logger.process_simulated(result)
            else:
                # not enough balance to sell
                pass
        if action == 1 and symbol == "BTC":
            # buy BTC
            usd_balance = prev_state["USD_Balance"]
            if usd_balance >= amount * prev_state["BTC_Price"]:
                if price is None:
                    print("BrokerAgent - place buy order at current price")
                    result = self.place_market_buy_order(symbol, amount)
                    self.pnl_agent.update_state()
                    self.sql_logger.process_simulated(result)
                else:
                    print("BrokerAgent - place buy order at current price")
                    result = self.place_limit_buy_order(symbol, amount, price)
                    self.pnl_agent.update_state()
                    self.sql_logger.process_simulated(result)
            else:
                # no enough balance - stop trading
                pass
        pass

    # can only make transaction at current price, fill is assumed to be full
    def advice_transaction(self, action, symbol, amount, prev_state):
        recc_cost = amount * prev_state["BTC_Price"]
        # print("BrokerAgent - recc_cost:", recc_cost)
        print("BrokerAgent balances {} {}".format(self.advice_balance_usd, self.advice_balance_btc))

        if action == -1 and symbol == "BTC":
        # sell BTC
            if self.advice_balance_btc > amount:
                print("BrokerAgent - advice sell order at current price")
                self.sql_logger.process_advice("sell", symbol, amount, prev_state["BTC_Price"])
                self.advice_balance_btc -= amount
                self.advice_balance_usd += recc_cost
                self.pnl_agent.update_state()
            else:
                # not enough balance to sell
                pass
        if action == 1 and symbol == "BTC":
        # buy BTC
            recc_cost = amount * prev_state["BTC_Price"]
            if self.advice_balance_usd > recc_cost:
                print("BrokerAgent - advice sell order at current price")
                self.sql_logger.process_advice("buy", symbol, amount, prev_state["BTC_Price"])
                self.advice_balance_btc += amount
                self.advice_balance_usd -= recc_cost
                self.pnl_agent.update_state()
            else:
                # not enough balance to buy
                pass

    # methods to call API
    def get_ohlcv_data(self, symbol, candlestick_timeframe, limit=100):
        """
        OHLCV data includes 5 data points:
        the Open and Close represent the first and the last price level during a specified interval.
        High and Low represent the highest and lowest reached price during that interval.
        Volume is the total amount traded during that period.
        """
        ohlcv = self.hitbtc.fetch_ohlcv(symbol, candlestick_timeframe, limit=limit,
                                               params={'sort': 'DESC'})
        df = pd.DataFrame(ohlcv, columns=['Date', 'open', 'high', 'low', 'close', 'volume'])
        # transform date
        df['date'] = df.Date.apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S'))
        df.index = df.set_index('date').index.astype('datetime64[ns]')
        return df

    def get_balance(self, currency):
        """
        Balance of currency
        currency is one of "USD", "BTC", "ETH"
        """
        balance = self.hitbtc.fetch_balance({'type': 'trading'})
        for k in balance['info']:
            if k['currency'] == currency:
                return k['available']
        return None

    def get_ticker_price(self, symbol):
        """
        get crypto market price
        """
        res = self.hitbtc.fetch_ticker(symbol)
        # The bid price refers to the highest price a buyer will pay for a security.
        # The ask price refers to the lowest price a seller will accept for a security.
        # ticker price = (bid + ask) / 2
        return ((float)(res['info']['bid']) + (float)(res['info']['ask'])) / 2

    def place_market_buy_order(self, symbol, amount):
        """
        place a buy order at market price
        """
        res = self.hitbtc.create_market_buy_order(symbol+"USD", amount)
        return res

    def place_market_sell_order(self, symbol, amount):
        """
        place a sell order at market price
        """
        res = self.hitbtc.create_market_sell_order(symbol+"USD", amount)
        return res

    def place_limit_buy_order(self, symbol, amount, price):
        """
        place a buy order at limit price
        """
        res = self.hitbtc.create_limit_buy_order(symbol+"USD", amount, price)
        return res

    def place_limit_sell_order(self, symbol, amount, price):
        """
        place a sell order at limit price
        """
        res = self.hitbtc.create_limit_sell_order(symbol+"USD", amount, price)
        return res

    def get_order_status(self, clientOrderId, symbol):
        """
        check status of an existing order
        """
        res = self.hitbtc.fetch_order(clientOrderId, symbol)
        return res['status']

    def cancel_order(self, clientOrderId, symbol):
        """
        cancel an order by order id
        """
        self.hitbtc.cancel_order(clientOrderId, symbol)


if __name__ == "__main__":
    pass
    # broker_agent = BrokerAgent()
    # print(broker_agent.get_balance("USD"))
    # print(broker_agent.get_ticker_price("BTCUSD"))
    # print(broker_agent.place_market_buy_order("BTCUSD", 0.001))
    # print(broker_agent.place_market_sell_order("BTCUSD", 0.001))
    # print(broker_agent.place_limit_buy_order("BTCUSD", 0.001, 30000))
    # print(broker_agent.place_limit_sell_order("BTCUSD", 0.001, 30000))
    # broker_agent.cancel_order("1cd86f958c014697a0e5c91412ba7c0a", "BTCUSD")
    # df = broker_agent.get_ohlcv_data("BTCUSD", "1m")
    # print(df.head())


