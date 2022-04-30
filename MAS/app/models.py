from datetime import datetime
import pytz
from app import db, Base
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from flask_login import UserMixin
from sqlalchemy import Column, Integer, Float, String, ForeignKey
import random

"""
This class contains the sql tables that will be used to
store to and call information from
"""

class SimulatedTransactions(UserMixin, Base):
    __tablename__ = "simulated_transactions"
    id = db.Column(db.String(255), primary_key=True)
    symbol = db.Column(db.String(255), nullable=False)
    action = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False, default=0.0)
    req_volume = db.Column(db.Float, nullable=False, default=0.0)
    actual_volume = db.Column(db.Float, nullable=False, default=0.0)
    total = db.Column(db.Float, nullable=False, default=0.0)
    fill = db.Column(db.String)
    time = db.Column(DateTime(timezone=True))

class AdviceTransactions(UserMixin, Base):
    __tablename__ = "advice_transactions"
    id = db.Column(db.String(255), primary_key=True)
    symbol = db.Column(db.String(255), nullable=False)
    action = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False, default=0.0)
    volume = db.Column(db.Float, nullable=False, default=0.0)
    total = db.Column(db.Float, nullable=False, default=0.0)
    time = db.Column(DateTime(timezone=True))


class RecordedSolutions(UserMixin, Base):
    __tablename__ = "recorded_solutions"
    id = db.Column(db.String(255), primary_key=True)
    symbol = db.Column(db.String(255), nullable=False)
    action = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False, default=0.0)
    volume = db.Column(db.Float, nullable=False, default=0.0)


