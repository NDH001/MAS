import requests
import json
import time
import numpy as np
from datetime import datetime, tzinfo
import pytz
import warnings

import random
import re
import random, string
from random import randrange

from sklearn.neighbors import KNeighborsRegressor

"""
CBR helper class
This will retrieve data from database and do the CBR (KNN)
"""
class CBR:
    def __init__(self, mode, db):
        # record transactions
        self.mode = mode
        self.database = db

    # find new solution based on prev solution
    def find_prev_solution(self, action, symbol, volume, current_price):
        # import database models
        from app.models import RecordedSolutions

        # get all prev solutions
        query = (self.database.session.query(RecordedSolutions)
                 .filter(RecordedSolutions.symbol == symbol)
                 .all())

        # if no solution, return none
        if query is None:
            return None

        action_arr = []
        price_arr = []
        volume_arr = []

        for row in query:
            action_arr.append(float(row.action))
            price_arr.append(float(row.price))
            volume_arr.append(float(row.volume))

        # use action,price,volume to get recommended action and volume
        feats = list(zip(action_arr, price_arr, volume_arr))
        labels = list(zip(action_arr, volume_arr))

        neigh = KNeighborsRegressor(n_neighbors=2)
        neigh.fit(feats, labels)

        # predict action and volume
        prediction = neigh.predict([(action, current_price, volume)])

        return prediction

    # store a solution
    def store_solution(self, symbol, action, price, volume):
        # import database models
        from app.models import RecordedSolutions

        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        tr = RecordedSolutions(id=random_id, symbol=symbol, action=action, price=price, volume=volume)

        self.database.session.add(tr)
        self.database.session.commit()
