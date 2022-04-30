from flask_wtf import FlaskForm
from wtforms import FloatField, RadioField, StringField, PasswordField, BooleanField, SubmitField, IntegerField, validators, SelectField, HiddenField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired

"""
This class contains the forms that will appear on the UI
"""

# control panel form
# allows user to input: Mode, starting balance transaction limit, max loss, risk percent
class ControlForm(FlaskForm):
    mode = RadioField("Mode: ", choices=[("Simulated", "Simulated"),("Advice_Only", "Advice Only")], default="Simulated")
    balance = FloatField('Starting balance (in USD):',
        [validators.DataRequired()],
        default = 2000
    )
    limit = FloatField('Limit for a single transaction (in USD):',
        [validators.DataRequired()],
        default = 100
    )

    max_loss = RadioField("Max loss: ", choices=[(10, "10%"), (20, "20%"), (30, "30%"), (40, "40%"), (50, "50%")], default=10)
    risk = RadioField("Risk: ", choices=[("Conservative", "Conservative"), ("Neutral", "Neutral"), ("Aggressive", "Aggressive")], default="Conservative")
    submit = SubmitField('Start')

# button to stop run
class ControlFormActive(FlaskForm):
    submit = SubmitField('Stop')
