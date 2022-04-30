from threading import Thread, Semaphore, Event
import requests
import json
import time
from datetime import datetime, tzinfo
import pytz
import warnings

import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()

"""
This class is a simple MLP
Uses data for past 72 hours to suggest next hour recommendation
"""
class MLPAgent:
    def __init__(self):
        self.mlp = None
        self.current_time = None
        self.current_date = None
        self.check_time()

        self.exit_flag = False
        self.sleep_time = 1800 # 30 mins
        self.thread = Thread(name=self.__str__(), target=self.run)

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
        # pass
        sem.acquire()
        self.check_time()
        print("MLPAgent - Advice:{}".format(self.get_advice_MLP()))
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('MLPAgent - THREAD TERMINATED!')

    def load_mlp(self):
        self.mlp = MLP()
        self.mlp.getAdviceMLP(train=True, debug=True)

    def get_advice_MLP(self):
        return self.mlp.getAdviceMLP()

    def check_time(self):
        # day
        utc_now_dt_str = datetime.now(tz=pytz.UTC).strftime("%m/%d/%Y")
        # hour
        utc_now_dt = datetime.now(tz=pytz.UTC)

        # train mlp
        if self.current_date is None:
            self.current_date = utc_now_dt_str
            print("MLPAgent - Loading MLP:", self.current_date)
            self.load_mlp()
            self.current_time = utc_now_dt.hour
        elif utc_now_dt_str != self.current_date:
            self.current_date = utc_now_dt_str
            print("MLPAgent - Getting MLP for day:", self.current_date)
            self.load_mlp()
            self.current_time = utc_now_dt.hour

        if self.current_time is None or utc_now_dt.hour != self.current_time:
            self.current_time = utc_now_dt.hour
            print("MLPAgent - Getting MLP value for hour:", self.current_time)
            self.mlp.getAdviceMLP()

# a class used to build the MLP model, the model takes in 72 data points input, has a 1024 hidden layer dimension,
# a drop out layer to prevent overfitting and lastly a regression output to predict the trends.
class MLPRegression(nn.Module):
    def __init__(self, dim=1024):
        super(MLPRegression, self).__init__()

        self.hidden1 = nn.Sequential(

            nn.Linear(72, dim),
            nn.Linear(dim, dim),
            nn.Dropout(p=0.2),
        )

        self.predict = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.hidden1(x)
        output = self.predict(x)
        return output[:, 0]


class MLP:
    #learning rate 0.01,batchsize = 10,criterion = MSE,inputs = 72 hours of closing price
    def __init__(self):
        self.lr = 0.01
        self.device = torch.device('cpu')
        self.net = MLPRegression()
        self.batch_size = 10
        self.net = self.net.to(self.device)
        self.criterion = nn.MSELoss()
        self.df = pd.DataFrame(yf.Ticker('BTC-USD').history(period='100d', interval='1h'))
        self.btc_df = self.df.drop(['Dividends', 'Stock Splits'], axis=1)
        self.hours = 72

    # preprocess clearning, we first retrieve the data we need from the close column, normalize them, so that they are within \n
    # reasonable scale, split them up according to train and test data, with former at 2000, and latter at around 138
    def EDA(self):
        # print((self.btc_df.tail()))
        training_set = self.btc_df.loc[:, ['Close']]
        training_set = training_set.values

        sc = MinMaxScaler(feature_range=(0, 1))
        training_set = sc.fit_transform(training_set)

        x_train = []
        y_train = []
        for i in range(self.hours, len(training_set)):
            x_train.append(training_set[i - self.hours:i])
            y_train.append(training_set[i, training_set.shape[1] - 1])
        x, y = np.array(x_train), np.array(y_train)

        x = torch.from_numpy(np.asarray(x).astype(np.float32))
        y = torch.from_numpy(np.asarray(y).astype(np.float32))
        x_train = x[:2000]
        y_train = y[:2000]
        x_test = x[2000:]
        y_test = y[2000:]

        return x_train, y_train, x_test, y_test, training_set


    # a eval function used to evaluate the accuracy of the model
    def eval_on_test_set(self, x_test, y_test):

        total_error = 0

        for i in range(0, 280, self.batch_size):
            data = x_test[i:i + self.batch_size]
            label = y_test[i:i + self.batch_size]

            data = data.to(self.device)
            label = label.to(self.device)

            inputs = data.view(self.batch_size, self.hours)


            scores = self.net(inputs)

            error = self.get_error(scores, label)

            total_error += error

        print(f'Total error on testing set : {total_error / (380 // self.batch_size)}')


    # a function to calculate the total error by setting up a error threshold, any error goes beyond 0.2, \n
    # be it negative or positive, are considered as wrong prediction.
    def get_error(self, scores, labels):

        bs = scores.size(0)
        matches = 0

        for i in range(len(scores)):
            if abs(scores[i] - labels[i]) <= 0.2:
                matches += 1

        return 1 - matches / self.batch_size

    # the main training function used to train the model, every 7th epoch, the learning rate would be reduced by 1.5 in a total of 40 epochs.
    # the batches are selected randomly to prevent memorization, and finally a debug function is added for debugging and visualization

    def train(self, x_train, y_train, x_test, y_test, debug=False):
        l_r = self.lr
        for epoch in range(1, 40):

            if epoch % 7 == 0:
                l_r = self.lr / 1.5

            optimizer = torch.optim.SGD(self.net.parameters(), lr=l_r)
            total_loss = 0
            total_error = 0

            random = torch.randperm(2000)

            for count in range(0, 2000, self.batch_size):
                optimizer.zero_grad()
                indices = random[count:count + self.batch_size]

                data = x_train[indices].to(self.device)

                label = y_train[indices].to(self.device)

                inputs = data.view(self.batch_size, self.hours)

                inputs.requires_grad_()
                predict = self.net(inputs)

                loss = self.criterion(predict, label)

                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()
                total_error += self.get_error(predict.detach(), label)

                if epoch == 39:
                    torch.save(self.net.state_dict(), 'MLP.pth')

            if debug:
                print(
                    f"Epoch: {epoch} Learning rate: {l_r} Total Loss: {total_loss / (2000 // self.batch_size)} Total Error: {total_error / (2000 // self.batch_size)}")
                self.eval_on_test_set(x_test, y_test)


    # a function to load the pretrained model from the directory. It is assumed that we would only be training the model every once \n
    # per hour, to speed up the decision making process, but at the same time, preserves accuracy and trustworthiness.
    def load_pretrain(self):
        self.net.load_state_dict(torch.load('MLP.pth'))
        self.net.eval()

    # a function to concatenate all the data points, be it train or testing data and plot it out \n
    # to see if the trend of prediction is following the actual market movement.
    # if show is true, this allows for debug and visualization of the most recent 72 hours market and prediction trend
    def result(self, x_train, x_test, training_set, show=True):

        predicted_test = self.net(x_test.squeeze().view(x_test.shape[0], self.hours).to(self.device))
        predicted_train = self.net(x_train.squeeze().view(x_train.shape[0], self.hours).to(self.device))

        temp = torch.zeros(self.hours)
        predicted = torch.cat([temp, predicted_train, predicted_test])

        self.btc_df['Norm_Close'] = training_set
        self.btc_df['Norm_Close_P'] = predicted.detach().numpy()

        if show:
            fig, ax = plt.subplots(1)
            plt.plot(self.btc_df.Norm_Close[len(self.btc_df)-self.hours:], label='Normalized Closing price')
            plt.plot(self.btc_df.Norm_Close_P[len(self.btc_df)-self.hours:], label='Normalized predicted Closing price')
            plt.xticks(rotation=25)
            plt.grid()
            plt.legend()
            plt.show()

    # a function that returns the final decision of either buy or sell based on the last predicted trend made by \n
    # the mlp, if the predicted price is higher, buy, otherwise, sell.

    def getAdviceMLP(self, train=False,debug = False):
        x_train, y_train, x_test, y_test, training_set = self.EDA()
        if train:
            self.train(x_train, y_train, x_test, y_test, debug=debug)
        else:
            self.load_pretrain()
        self.result(x_train, x_test, training_set)
        mask = self.btc_df.Norm_Close_P > self.btc_df.Norm_Close
        return 'buy' if mask[-1] == True else 'sell'
