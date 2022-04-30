from flask import Flask, url_for, redirect, render_template, request, abort, send_from_directory, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import exists, func, update
from sqlalchemy.orm import aliased

import random
import re
from sqlalchemy import desc, asc
from datetime import datetime
import pytz
import random, string
from random import randrange
import datetime
from app.agents.agent_controller import Agent_controller

"""
This class is where all the magic happens
Flask uses routing, so different pages can be opened
"""

# create the app
app = Flask(__name__)
app.config.from_pyfile('config.py')

# create the controller
controller_active = False
# mode, balance, limit, max loss, risk
controller_params = [None, None, None, None, None]
controller = Agent_controller()

# create the database, login with credentials
db = SQLAlchemy(app)


# base database table model
class Base(db.Model):
    __abstract__ = True

    def add(self):
        try:
            db.session.add(self)
            self.save()
        except:
            db.session.rollback()

    def save(self):
        try:
            db.session.commit()
        except:
            db.session.rollback()

    def delete(self):
        try:
            db.session.delete(self)
            self.save()
        except:
            pass


# import database models
from app.models import AdviceTransactions, SimulatedTransactions, RecordedSolutions
db.create_all()

# import forms
from app.forms import ControlForm, ControlFormActive


# home
@app.route('/')
def index():
    weburl = "https://app.powerbi.com/reportEmbed?reportId=a653d52f-fd6d-463d-a72d-d236588add31&autoAuth=true&ctid=5ba5ef5e-3109-4e77-85bd-cfeb0d347e82&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLXNvdXRoLWVhc3QtYXNpYS1yZWRpcmVjdC5hbmFseXNpcy53aW5kb3dzLm5ldC8ifQ%3D%3D"

    # query the database
    query = None
    if controller_params[0] is not None:
        if controller_params[0] == "Simulated":
            query = (db.session.query(SimulatedTransactions)
                 .order_by(SimulatedTransactions.time.desc())
                 .limit(10)
                 .all())
        else:
            query = (db.session.query(AdviceTransactions)
                     .order_by(AdviceTransactions.time.desc())
                     .limit(10)
                     .all())

    return render_template('index.html', rows=query, page='home', weburl=weburl, active=controller_active,
                           mode=controller_params[0])


@app.route('/dashboard')
def dashboard():
    return index()

@app.route('/control_panel', methods=["GET", "POST"])
def control_panel():
    # not active
    control_form = ControlForm()
    # active
    control_form_active = ControlFormActive()
    if request.method == 'POST':
        global controller_active
        global controller
        global controller_params

        if controller_active:
            controller.deactivate()
            controller_active = False
            controller_params = [None, None, None, None, None]
            return render_template('control_panel.html', page='control', form=control_form, active=controller_active, par=controller_params)
        else:
            mode = request.form['mode']
            balance = request.form['balance']
            limit = request.form['limit']
            max_loss = request.form['max_loss']
            risk = request.form['risk']

            # delete advices table
            scrub()

            # mode, init balance, limit, max loss, risk
            controller_params = [mode, balance, limit, max_loss, risk]
            controller.activate(controller_params, db)
            controller_active = True


            return render_template('control_panel.html', page='control',  form=control_form_active, active=controller_active, par=controller_params)
    else:
        if controller_active:
            return render_template('control_panel.html', page='control',  form=control_form_active, active=controller_active, par=controller_params)
        else:
            return render_template('control_panel.html', page='control',  form=control_form, active=controller_active, par=controller_params)

# Faq page
@app.route('/faq')
def faq():
    return render_template('faq.html', page='faq')

# delete all transactions for advice mode
# this ensures 'advice mode' will only have advice for that run
@app.route('/scrubTransactions')
def scrub():
    models.AdviceTransactions.query.delete()
    db.session.query()

    db.session.commit()
    return ('transactions deleted')

"""
Utility methods.
Uncomment to use these methods
"""

# create some mock data
# @app.route('/create')
# def create():
#     # Mock transaction data
#     mock_data = []
#     for i in range(0,10):
#         random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
#         random_action = random.choice(["buy", "sell"])
#         random_symbol = random.choice(["BTC", "ETH"])
#         random_price = round(random.uniform(33.33, 66.66), 2)
#         random_vol = round(random.uniform(33.33, 66.66), 2)
#         random_fill = random.choice(["full","partial"])
#
#         startDate = datetime.datetime(2022, 2, 20, 13, 00)
#         random_timestamp = startDate + datetime.timedelta(minutes=randrange(60))
#
#         mock_data_val = [random_id,random_symbol,random_action,random_price,random_vol,random_fill,random_timestamp]
#         mock_data.append(mock_data_val)
#
#     for i in mock_data:
#         u = Transactions(id=i[0],symbol=i[1],action=i[2],price=i[3],volume=i[4],fill=i[5],time=i[6])
#         db.session.add(u)
#     db.session.commit()
#     return ('mock transactions added')

# create some cbr solutions
# @app.route('/create_sols')
# def create_sols():
#     # Mock transaction data
#     mock_data = []
#     for i in range(0, 5):
#         random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
#         random_action = random.choice([1, -1])
#         random_symbol = random.choice(["BTC", "ETH"])
#         random_price = round(random.uniform(4000, 5000), 2)
#         random_vol = round(random.uniform(0.1, 1.0), 2)
#
#         mock_data_val = [random_id, random_symbol, random_action, random_price, random_vol]
#         mock_data.append(mock_data_val)
#
#     for i in mock_data:
#         u = RecordedSolutions(id=i[0], symbol=i[1], action=i[2], price=i[3], volume=i[4])
#         db.session.add(u)
#     db.session.commit()
#
#     return ('mock solutions added')



