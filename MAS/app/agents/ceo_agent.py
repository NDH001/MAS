from threading import Thread, Semaphore, Event
import requests
import json
import time
from datetime import datetime, tzinfo
import pytz
import warnings

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()
"""
CEO Agent 
Is the handbrake to stop the MAS based on certain conditions
"""
class CEOAgent:
    def __init__(self, controller_params, agent_controller):
        self.agent_controller = agent_controller
        # mode, init balance, limit, max loss, risk
        self.mode = controller_params[0]
        self.orig_balance = controller_params[1]
        self.limit = controller_params[2]
        self.max_loss = controller_params[3]
        self.risk = controller_params[4]

        self.final_decision = None
        self.pnl_agent = None
        self.broker_agent = None
        self.decider_agent = None

        self.exit_flag = False
        self.sleep_time = 1800 # 30 mins
        self.thread = Thread(name=self.__str__(), target=self.run)

    def init_agents(self, p_agent, br_agent, dr_agent):
        # set agents
        self.pnl_agent = p_agent
        self.broker_agent = br_agent
        self.decider_agent = dr_agent
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
        self.calculate_final_decision()
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('CEOAgent - THREAD TERMINATED!')

    def calculate_final_decision(self):
        print("CEOAgent - Making final decision")

        # get full info from PNLAgent
        prev_state = self.pnl_agent.get_prev_state()
        # print('pnl prev state:', prevState)

        # get decision from DeciderAgent
        decision, amount, buy_agents, sell_agents, buy_weight, sell_weight, advs = self.decider_agent\
            .get_final_suggestion()

        # convert USD to BTC or ETH
        amount = amount/float(prev_state["BTC_Price"])

        print("Decision - dec:{}, amt:{}, ba:{},sa:{},bw:{},sw:{},adv:{} ".format(decision, amount, buy_agents,
                                                                                  sell_agents, buy_weight,
                                                                                  sell_weight, advs))
        symbol = "BTC"

        usd_balance = float(prev_state["USD_Balance"])
        btc_balance = float(prev_state["BTC_Balance"])
        eth_balance = float(prev_state["ETH_Balance"])

        curr_btc_price = float(prev_state["BTC_Price"])
        curr_eth_price = float(prev_state["ETH_Price"])

        print("CEOAgent - USD balance:{}, BTC balance:{}, ETH balance:{}".
              format(usd_balance, btc_balance, eth_balance, amount))

        # handbrake
        # if too much loss
        account_value = usd_balance + (curr_btc_price*btc_balance) + (curr_eth_price*eth_balance)
        print("CEOAgent debug - acceptable loss: ", float(self.max_loss)/100 * float(self.orig_balance))
        print("CEOAgent debug - current loss:", float(self.orig_balance) - account_value)

        if (float(self.max_loss) * float(self.orig_balance)) <= float(self.orig_balance) - account_value:
            # stop trading
            self.agent_controller.deactivate()
            self.final_decision = [-1, "BTC", btc_balance]
        if curr_btc_price > prev_state["BTC_Take_Profit"] or curr_btc_price < prev_state["BTC_Stop_Loss"]:
            self.final_decision = [-1, "BTC", btc_balance]
        else:
            self.final_decision = [decision, symbol, amount]


        print("CEOAgent - Final decision:{}, symbol: {}, amount:{}USD.".format(decision, symbol, amount))

    def get_final_decision(self):
        if self.final_decision is None:
            self.calculate_final_decision()
        return self.final_decision


if __name__ == "__main__":
    cp = [None, None, None, None, None]
    broker_agent = BrokerAgent()
    pnl_agent = PNLAgent(broker_agent)
    decider_agent = DeciderAgent()

    ceo = CEOAgent(cp, pnl_agent, broker_agent, decider_agent)
