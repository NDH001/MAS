from app.agents.MA_agent import MovingAverageAgent
from app.agents.BB_agent import BollingerBandAgent
from app.agents.pair_agent import PairAgent
from app.agents.MLP_agent import MLPAgent
from app.agents.MLP_agent import MLPAgent
from app.agents.Qualitative_agent import QualitativeAgent

from app.agents.broker_agent import BrokerAgent
from app.agents.pnl_agent import PNLAgent
from app.agents.decider_agent import DeciderAgent
from app.agents.ceo_agent import CEOAgent
from app.agents.sql_logger import SQLLogger
from app.agents.cbr import CBR

from threading import Thread
import requests
import json
import time
import os
import sys

"""
This class controls all the initialization and shutdown of agents
"""
class Agent_controller:
    def __init__(self):
        self.active = False

        self.moving_avg_agent = None
        self.bollinger_band_agent = None
        self.pair_agent = None
        self.mlp_agent = None
        self.quali_agent = None

        self.sql_logger = None
        self.cbr = None

        self.broker_agent = None
        self.pnl_agent = None
        self.decider_agent = None

        self.ceo_agent = None

    def __del__(self):
        print("I'm being automatically shutdown. Goodbye!")

    # activate all agents
    def activate(self, controller_params, db):
        mode = controller_params[0]
        balance = controller_params[1]
        limit = controller_params[2]
        max_loss = controller_params[3]
        risk = controller_params[4]

        self.active = True
        # N-agents
        self.moving_avg_agent = MovingAverageAgent()
        self.bollinger_band_agent = BollingerBandAgent()
        self.pair_agent = PairAgent()
        self.mlp_agent = MLPAgent()
        self.quali_agent = QualitativeAgent()

        # others
        self.sql_logger = SQLLogger(mode, db)
        self.cbr = CBR(mode, db)
        self.broker_agent = BrokerAgent(mode, balance, self.sql_logger)

        self.pnl_agent = PNLAgent(mode)
        self.decider_agent = DeciderAgent(limit, risk, self.cbr)

        # handbrake
        self.ceo_agent = CEOAgent(controller_params, self)

        self.init_agents()

    # init reference to other agents
    def init_agents(self):
        self.decider_agent.init_agents(self.moving_avg_agent, self.bollinger_band_agent, self.mlp_agent, self.pair_agent, self.quali_agent,
                                        self.pnl_agent)

        self.ceo_agent.init_agents(self.pnl_agent, self.broker_agent, self.decider_agent)
        self.broker_agent.init_agents(self.pnl_agent, self.ceo_agent)

        # PNL needs broker to be initiated
        self.pnl_agent.init_agents(self.broker_agent, self.decider_agent)

        self.moving_avg_agent.start_threads()
        self.bollinger_band_agent.start_threads()
        self.pair_agent.start_threads()
        self.mlp_agent.start_threads()
        self.quali_agent.start_threads()

        self.broker_agent.start_threads()
        self.pnl_agent.start_threads()
        self.decider_agent.start_threads()
        self.ceo_agent.start_threads()

    # deactivate all agents
    def deactivate(self):
        print("deactiving MAS")
        self.active = False
        self.moving_avg_agent.terminate()
        self.bollinger_band_agent.terminate()
        self.pair_agent.terminate()
        self.mlp_agent.terminate()
        self.quali_agent.terminate()

        self.broker_agent.terminate()
        self.pnl_agent.terminate()
        self.decider_agent.terminate()
        self.ceo_agent.terminate()
        print("MAS deactivated")
