from threading import Thread, Semaphore, Event
from datetime import datetime
import threading as th
import requests
import json
import time
import warnings

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()

"""
This agent will send the latest data to PowerBI
"""
POWERBI_URL = "https://api.powerbi.com/beta/5ba5ef5e-3109-4e77-85bd-cfeb0d347e82/datasets/1fe0159f-42df-460a-a206-2f2bc66171fd/rows?key=Kl0%2BpSk1UB1VYUDjD8JLp83K5BeC59kCoriOiPII0tcRJHPUtJKxHhMlF5NKOREEKHbTSVt%2BDL7NmzWYHcRvgg%3D%3D"
# fix rate for take profit and stop loss
TAKE_PROFIT_RATE = 1.1
STOP_LOSS_RATE = 0.9

class PNLAgent:
    def __init__(self, mode):
        self.mode = mode
        self.broker_agent = None
        self.decider_agent = None

        self.prevState = None

        self.exit_flag = False
        self.sleep_time = 30 # every 30 secs
        self.thread = Thread(name=self.__str__(), target=self.run)

    def init_agents(self, br_agent, dr_agent):
        # set agents
        self.broker_agent = br_agent
        self.decider_agent = dr_agent
        # any req methods
        self.init_first_state()

    def start_threads(self):
        # start thread
        self.thread.start()

    def run(self):
        while True:
            self.tick()
            event_obj.wait(self.sleep_time)
            if self.exit_flag:
                break

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('pnlAgent - THREAD TERMINATED!')

    def tick(self):
        sem.acquire()
        self.update_prices()
        # send data to powerbi
        if self.mode == "Simulated":
            data = [self.prevState]
            headers = {
                "Content-Type": "application/json"
            }
            # Send data to PowerBI Here
            response = requests.request(
                method="POST",
                url=POWERBI_URL,
                headers=headers,
                data=json.dumps(data)
            )
        sem.release()

    # first initialization, set some values to 0 (e.g. balance)
    def init_first_state(self):
        usd_balance, btc_balance, eth_balance, btc_price, eth_price = self.broker_agent.fetch_data()

        # pnl agent always record previous state of trade system, update in each tick
        self.prevState = {
            "Datetime": str(datetime.now()),
            "BTC_Price": btc_price,
            "ETH_Price": eth_price,
            "BTC_Balance": btc_balance,
            "ETH_Balance": eth_balance,
            "USD_Balance": usd_balance,
            "BTC_Action": 0,  # 0 -> no action, 1 -> buy, -1 -> sell
            "BTC_Amount": 0,
            "ETH_Action": 0,  # 0 -> no action, 1 -> buy, -1 -> sell
            "ETH_Amount": 0,
            # Floating Profit or Loss is the profit or loss that a trader has when they hold an open position.
            # It floats (changes) since it changes in correspondence with the open position(s).
            # trader can keep track of how their open positions are doing and see when he should close them
            "Floating_Profit": 0,
            # take profit: tells broker how much you are willing to make as a profit with one trade and close
            # it once you’re happy with the amount.
            "BTC_Take_Profit": btc_price * 1.1,
            # stop loss: tells broker know how much you are willing to risk with your trade.
            "BTC_Stop_Loss": btc_price * 0.9,
            "ETH_Take_Profit": eth_price * 1.1,
            "ETH_Stop_Loss": eth_price * 0.9,
            # position = (value of crypto) / (net value of account)
            "Position": btc_balance * btc_price + eth_balance * eth_price /
                        (btc_balance * btc_price + eth_balance * eth_price + usd_balance),
            # average cost = market price if crypto balance is zero
            "BTC_Average_Cost": btc_price,
            "ETH_Average_Cost": eth_price,
            # account value = sum of [currency balance * currency price (in usd)]
            "Account_Value": usd_balance + btc_balance * btc_price + eth_balance * eth_price
        }

    # update prices only
    def update_prices(self):
        usd_balance, btc_balance, eth_balance, btc_price, eth_price = self.broker_agent.fetch_data()
        self.prevState["BTC_price"] = btc_price
        self.prevState["ETH_Price"] = eth_price

    # update all fields
    def update_state(self):
        usd_balance, btc_balance, eth_balance, btc_price, eth_price = self.broker_agent.fetch_data()

        # infer action from change of crypto balance
        btc_amount = abs(btc_balance - self.prevState["BTC_Balance"])
        eth_amount = abs(btc_balance - self.prevState["ETH_Balance"])
        # if balance not change -> no action
        # if balance increase -> buy  (current balance - previous balance)
        # if balance decrease -> sell (previous balance - current balance)
        btc_action = 0 if btc_amount == 0 else (btc_balance - self.prevState["BTC_Balance"]) / btc_amount
        eth_action = 0 if eth_amount == 0 else (eth_balance - self.prevState["ETH_Balance"]) / eth_amount

        account_value = usd_balance + btc_balance * btc_price + eth_balance * eth_price
        # update floating profit
        # current floating profit = previous floating profit + change of account value in current tick
        floating_profit = self.prevState["Floating_Profit"] + account_value - self.prevState["Account_Value"]

        # calculate average cost of btc and eth
        if btc_balance == 0:
            btc_average_cost = btc_price
        else:
            # update average cost (Weighted average)
            # average cost = (prev average cost * prev balance + buy/sell amount * current price) / (current balance * current price)
            btc_average_cost = (self.prevState["BTC_Average_Cost"] * self.prevState[
                "BTC_Balance"] + btc_action * btc_amount * btc_price) / (btc_balance * btc_price)

        if eth_balance == 0:
            eth_average_cost = eth_price
        else:
            # update average cost
            eth_average_cost = (self.prevState["ETH_Average_Cost"] * self.prevState[
                "ETH_Balance"] + eth_action * eth_amount * eth_price) / (eth_balance * eth_price)

        # take profit = Average cost * take profit rate
        # stop loss  = Average cost * stop loss rate
        btc_take_profit = btc_average_cost * TAKE_PROFIT_RATE
        btc_stop_loss = btc_average_cost * STOP_LOSS_RATE
        eth_take_profit = eth_average_cost * TAKE_PROFIT_RATE
        eth_stop_loss = eth_average_cost * STOP_LOSS_RATE
        # update position
        position = (btc_balance * btc_price + eth_balance * eth_price) / account_value

        self.prevState = {
            "Datetime": str(datetime.now()),
            "BTC_Price": btc_price,
            "ETH_Price": eth_price,
            "BTC_Balance": btc_balance,
            "ETH_Balance": eth_balance,
            "USD_Balance": usd_balance,
            "BTC_Action": btc_action,  # 0 -> no action, 1 -> buy, -1 -> sell
            "BTC_Amount": btc_amount,
            "ETH_Action": eth_action,  # 0 -> no action, 1 -> buy, -1 -> sell
            "ETH_Amount": eth_amount,
            "Floating_Profit": floating_profit,
            "BTC_Take_Profit": btc_take_profit,
            "BTC_Stop_Loss": btc_stop_loss,
            "ETH_Take_Profit": eth_take_profit,
            "ETH_Stop_Loss": eth_stop_loss,
            "Position": position,
            "BTC_Average_Cost": btc_average_cost,
            "ETH_Average_Cost": eth_average_cost,
            "Account_Value": account_value
        }

    def get_prev_state(self):
        return self.prevState


if __name__ == "__main__":
    broker_agent = BrokerAgent()
    pnl_agent = PNLAgent(broker_agent)
