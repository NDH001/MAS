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

"""
SQL helper class
This will write data of transactions to database
"""
class SQLLogger:
    def __init__(self, mode, db):
        # record transactions
        self.mode = mode
        self.database = db

    # for simulated run, store data
    def process_simulated(self, result):
        from app.models import SimulatedTransactions

        if result is not None:
            transaction_time = datetime.now(tz=pytz.UTC)
            # transaction = "buy" if action == 1 else "sell" if action == -1 else "no action"

            fill = "full" if result["remaining"] == 0.0 else "partial"
            tr = SimulatedTransactions(id=result["id"], symbol=result["symbol"][:-5], action=result["side"],
                                       price=result["price"], req_volume=result["amount"], actual_volume=result["filled"],
                                       total=result["cost"], fill=fill, time=transaction_time)

            self.database.session.add(tr)
            self.database.session.commit()

    # for advice run, store data
    def process_advice(self, action, symbol, amount, price):
        # import database models
        from app.models import AdviceTransactions

        # transaction time and id
        transaction_time = datetime.now(tz=pytz.UTC)
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        transaction = "buy" if action == 1 else "sell" if action == -1 else "no go"
        total = float(amount)*float(price)
        tr = AdviceTransactions(id=random_id, symbol=symbol, action=transaction, price=price, volume=amount,
                                   total=total, time=transaction_time)

        # id, symbol, price volume, fill, time
        self.database.session.add(tr)
        self.database.session.commit()
