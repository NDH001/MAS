from threading import Thread, Semaphore, Event
import requests
import json
import time
import numpy as np
from datetime import datetime, tzinfo
import pytz
import warnings

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()
"""
This class gets signals from n-agents
And determines an action and volume of BTC to buy
It then checks CBR to predict a better final output
(based on past solutions)
"""
class DeciderAgent:
    def __init__(self, limit, risk, cbr):

        self.init = False
        self.adv = {}
        self.limit = float(limit)  # limit per transaction
        self.risk = 0.5 if risk == "Conservative" else 0.7 if risk == "Neutral" else 0.9

        self.ma_agent = None
        self.bb_agent = None
        self.mlp_agent = None
        self.pair_agent = None
        self.Qualitative_agent = None
        self.cbr = cbr
        self.prev_price = None

        # sends back data to decider
        self.pnl_agent = None

        self.current_time = None
        self.final_decision = None
        self.check_time()

        self.exit_flag = False
        self.sleep_time = 1800 # 30min
        self.thread = Thread(name=self.__str__(), target=self.run)

    # a function to initialize all the available agents
    def init_agents(self, ma_agent, bb_agent, mlp_agent, pair_agent, qualitative_agent, pnl_agent):
        # set agents
        self.ma_agent = ma_agent
        self.bb_agent = bb_agent
        self.mlp_agent = mlp_agent
        self.pair_agent = pair_agent
        self.Qualitative_agent = qualitative_agent
        self.pnl_agent = pnl_agent
        # any req methods

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
        self.check_time()
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('DeciderAgent - THREAD TERMINATED!')

    def check_time(self):
        # Get current datetime in UTC
        utc_now_dt = datetime.now(tz=pytz.UTC)
        if self.current_time is None:
            self.current_time = utc_now_dt.hour
            print("DeciderAgent - Getting decision for hour:", self.current_time)
            self.init = True
        elif utc_now_dt.hour != self.current_time:
            self.current_time = utc_now_dt.hour

            # adjust weights
            decision, amount, buy_agents, sell_agents, buy_weight, sell_weight, advs = self.get_final_suggestion()
            self.eval(decision, self.prev_price)

            # calculate new decision
            print("DeciderAgent - Getting decision for hour:", self.current_time)
            self.calc_final_suggestion()

    # a private function that assigns the initial weights to all 6 agents, we want it to be private so others can not make changes to it
    def __init_weight(self):
        return 100 / 6

    # get all agents' advice,and their weight and store them in a dictionary. Based on if this is the first time, the dictionary will either \n
    # update itself or initialize first.
    def get_all_advice(self):
        adv_dic = {}

        if self.init:
            init_weight = self.__init_weight()
            adv_dic['sma'] = [self.ma_agent.getAdviceSMA(), init_weight]
            adv_dic['ema'] = [self.ma_agent.getAdviceEMA(), init_weight]
            adv_dic['bb'] = [self.bb_agent.get_Advice_BB(), init_weight]
            adv_dic['pair'] = [self.pair_agent.getAdvicePair(), init_weight]
            adv_dic['mlp'] = [self.mlp_agent.get_advice_MLP(), init_weight]
            quali = self.Qualitative_agent.getAdviceQuali()
            adv_dic['quali'] = [quali, init_weight]
            self.init = False
        else:
            adv_dic['sma'] = [self.ma_agent.getAdviceSMA(), self.adv['sma'][1]]
            adv_dic['ema'] = [self.ma_agent.getAdviceEMA(), self.adv['ema'][1]]
            adv_dic['bb'] = [self.bb_agent.get_Advice_BB(), self.adv['bb'][1]]
            adv_dic['pair'] = [self.pair_agent.getAdvicePair(), self.adv['pair'][1]]
            adv_dic['mlp'] = [self.mlp_agent.get_advice_MLP(), self.adv['mlp'][1]]

            # quali = self.Qualitative_agent.getAdviceQuali()
            quali = 'sell'
            adv_dic['quali'] = [quali, self.adv['quali'][1]]
        return adv_dic

    # outputs a final decision based on the weightage of various agents and their suggestions.
    # the logic work as follows:
    # outputs the decision that has the highest weightage, for example, buy at 60%
    # we will perform buy action, but only at the ratio of those agents suggested, for example 2 out 6 agent suggested buy \n
    # then the ratio is 1/3
    # those past final decision would be store and used later again for cbr evaluation by using the KNN algorithm
    def calc_final_suggestion(self):
        buy_weight = 0
        sell_weight = 0

        buy_agents = []
        sell_agents = []
        advs = self.get_all_advice()

        decision = ''

        amount = 0

        # split agents to buy and sell groups
        for key in advs:
            if advs[key][0] == 'buy':
                buy_agents.append(key)
                buy_weight += advs[key][1]
            elif advs[key][0] == 'sell':
                sell_agents.append(key)
                sell_weight += advs[key][1]

        # if more buy than sell
        if buy_weight >= sell_weight:
            decision = 1
            percent = len(buy_agents)/6
            # calculate amount based on risk, limit and n-agents
            amount = self.risk * self.limit * percent
        else:
            decision = -1
            percent = len(sell_agents)/6
            # calculate amount based on risk, limit and n-agents
            amount = self.risk * self.limit * percent

        self.adv = advs

        self.final_decision = [decision, amount, buy_agents, sell_agents, buy_weight, sell_weight, advs]

        prev_state = self.pnl_agent.get_prev_state()
        cbr_prediction = self.cbr.find_prev_solution(decision, "BTC", amount, prev_state["BTC_Price"])

        # store solutions
        if cbr_prediction is None:
            self.cbr.store_solution("BTC", decision, prev_state["BTC_Price"], amount)
        else:
            self.cbr.store_solution("BTC", cbr_prediction[0][0], prev_state["BTC_Price"], cbr_prediction[0][1])
            self.final_decision[0] = cbr_prediction[0][0]
            self.final_decision[1] = cbr_prediction[0][1]

        # update latest price
        self.prev_price =  prev_state["BTC_Price"]

    # function to get the final suggestion
    def get_final_suggestion(self):
        if self.final_decision is None:
            self.calc_final_suggestion()
        return self.final_decision

    # After receiving the report from pnl agent, the system would do a evaluation based on the actions and result. \n
    # penalize those agents with wrong suggestions and increment the agents with right suggestions.
    # for example, pnl reflect a gain in profit with action buy, thus, all agents outputed buy would have a increment on their \n
    # weightage and all agents outputed sell would have a decrement on their weightage.
    def eval(self, action, prev_price):
        prev_state = self.pnl_agent.get_prev_state()
        _, buy_agents, sell_agents, _, _, advs = self.get_final_suggestion()
        current_price = prev_state["BTC_Price"]

        # if buy
        if action == 1:
            if current_price >= prev_price:
                for i in range(len(buy_agents)):
                    advs[buy_agents[i]] = ['buy', advs[buy_agents[i]][1] + 1]
            else:
                for i in range(len(buy_agents)):
                    advs[buy_agents[i]] = ['buy', advs[buy_agents[i]][1] - 1]
        # else sell
        elif action == -1:
            if current_price >= prev_price:
                for i in range(len(sell_agents)):
                    advs[sell_agents[i]] = ['sell', advs[sell_agents[i]][1] + 1]
            else:
                for i in range(len(sell_agents)):
                    advs[sell_agents[i]] = ['sell', advs[sell_agents[i]][1] - 1]

        self.adv = advs
