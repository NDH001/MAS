from threading import Thread, Semaphore, Event
import requests
import json
import time
from datetime import datetime, tzinfo
import pytz
import warnings

import tweepy
import configparser
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from google.cloud import language_v1
import os

warnings.filterwarnings("ignore")
event_obj = Event()
sem = Semaphore()

"""
This class gets tweets and determines the sentiments for the tweets
Based on sentiment, provide a buy/sell suggestion
"""

class QualitativeAgent:
    def __init__(self):
        self.btc_senti_data = None
        self.api = None
        self.auth = None
        self.client = None
        self.type_ = None
        self.encoding_type = None
        self.current_time = None
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
        sem.release()

    def terminate(self):
        self.exit_flag = True
        event_obj.set()
        print('QualitativeAgent - THREAD TERMINATED!')

    def check_time(self):
        # Get current datetime in UTC
        utc_now_dt = datetime.now(tz=pytz.UTC)

        if self.current_time is None or utc_now_dt.hour != self.current_time:
            self.current_time = utc_now_dt.hour
            print("QualitativeAgent - Getting sentiments for hour:", self.current_time)
            self.get_sentiment()

    # initialize twitter api
    def init_tweepy(self):
        config = configparser.ConfigParser()
        config.read('app/agents/config.ini')
        API_Key = config['twitter']['API_key']
        API_Key_Secret = config['twitter']['API_Key_Secret']
        Access_Token = config['twitter']['Access_Token']
        Access_Token_Secret = config['twitter']['Access_Token_Secret']

        self.auth = tweepy.OAuthHandler(API_Key, API_Key_Secret)
        self.auth.set_access_token(Access_Token, Access_Token_Secret)

        self.api = tweepy.API(self.auth)

    # initialize GCP natural language api
    def init_language(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'app/agents/static-pottery-304408-efb32334ffb2.json'
        self.client = language_v1.LanguageServiceClient()
        self.type_ = language_v1.Document.Type.PLAIN_TEXT
        self.encoding_type = language_v1.EncodingType.UTF8

    # The steps to get sentiments are as below:
    # 1. initiate the tweepy api
    # 2. initiate the google cloud sentiment api
    # 3. search for the most popular and recent btc related tweets on twitter
    # 4. pass in the tweets to sentiment api and calculates the strength, magnitude and score of these tweets
    # 5. computes the mean of the sentiments daily or monthly wise
    def get_sentiment(self):
        self.init_tweepy()
        self.init_language()

        date_time = []
        text = []
        public_tweets = self.api.search_tweets('btc',lang='en',result_type='popular')
        
        for tweets in public_tweets:
            date_time.append(tweets.created_at)
            text.append(tweets.text)
            
        info = {'timestamp':date_time,'tweets':text}
        btc_senti = pd.DataFrame(data=info)
        
        all_score =[]
        all_mag = []
        for i in range(len(btc_senti)):
            document = {"content": btc_senti.tweets.iloc[i], "type_": self.type_}
            response = self.client.analyze_sentiment(request = {'document': document, 'encoding_type': self.encoding_type})
            score = []
            magnitude = []
            for sentence in response.sentences:
                score.append(sentence.sentiment.score)
                magnitude.append(sentence.sentiment.magnitude)
            all_score.append(np.asarray(score).mean())
            all_mag.append(np.asarray(magnitude).mean())

        btc_senti['score'] = all_score
        btc_senti['mag'] = all_mag

        self.btc_senti_data = btc_senti
        
        self.btc_senti_data.timestamp = self.btc_senti_data.timestamp.astype(str)
        self.btc_senti_data['date'] = self.btc_senti_data['timestamp'].str.split(' ').str[0]
        self.btc_senti_data.date = self.btc_senti_data.date.map(lambda x: x.lstrip('2022-').rstrip(''))
        self.btc_senti_data= self.btc_senti_data.drop(['timestamp'],1)
        
        columns_titles = ['date','score','mag']
        self.btc_senti_data = self.btc_senti_data.reindex(columns=columns_titles)
        self.btc_senti_data = self.btc_senti_data.sort_values(by ='date' )
        
        btc_result = self.cryptoFilter(np.asarray(self.btc_senti_data))
        
        self.avg_per_day, self.detailed_per_day = self.compute_daily_average(self.btc_senti_data, btc_result)

    # function used to store all final gradings of the tweets
    def cryptoFilter(self,data):
        myResult = []
        for i in range(len(data)):
            myResult.append(self.fuzzy_logic(list(data[i])))
        return myResult
    
    # fuzzy logic algorithm that computes the grade of a particular tweet based on its score,magnitude and strength.
    # rules are specific from line 175 to 186
    def fuzzy_logic(self,data):
    
        curScore = data[1] 
    
        curMag = data[2]
    
        
        score = ctrl.Antecedent(np.arange(-1, 2, 1), 'score')
        mag = ctrl.Antecedent(np.arange(0, 11, 1), 'mag')
        strength = ctrl.Consequent(np.arange(0, 101, 1), 'strength')
    
        score.automf(3)
        mag.automf(3)
    
        strength['poor'] = fuzz.trimf(strength.universe, [0, 0, 50])
        strength['average'] = fuzz.trimf(strength.universe, [0, 50, 100])
        strength['good'] = fuzz.trimf(strength.universe, [50, 100, 100])
      
    
        rule1 = ctrl.Rule(score['poor'] & mag['poor'], strength['poor'])
        rule2 = ctrl.Rule(score['poor']& mag['average'], strength['poor'])
        rule3 = ctrl.Rule(score['poor']& mag['good'], strength['average'])
        
        
        rule4 = ctrl.Rule(score['average'] & mag['poor'], strength['poor'])
        rule5 = ctrl.Rule(score['average']& mag['average'], strength['average'])
        rule6 = ctrl.Rule(score['average']& mag['good'], strength['average'])
        
        rule7 = ctrl.Rule(score['good'] & mag['poor'], strength['poor'])
        rule8 = ctrl.Rule(score['good']& mag['average'], strength['average'])
        rule9 = ctrl.Rule(score['good']& mag['good'], strength['good'])
    
    
    
        investment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5,rule6,rule7,rule8,rule9])
        investment = ctrl.ControlSystemSimulation(investment_ctrl)
    
    
        investment.input['score'] = curScore
        investment.input['mag'] = curMag
    
    
        investment.compute()
    
        tempstrength = investment.output['strength']
        tempResult = tempstrength
    
    
        return tempResult
    
    
    # a function used to compute the average grade for all tweets retrieved on that day
    def compute_daily_average(self,temp_data,result):
        all_daily_sum = []
        daily_average = []
        daily_sum = []
        current_day = temp_data.date.iloc[0]
    
        for i in range(len(temp_data)):
            if current_day != temp_data.date.iloc[i]:
    
                current_day = temp_data.date.iloc[i]
                daily_average.append(np.mean(daily_sum))
                all_daily_sum.append(daily_sum)
                daily_sum = []
                daily_sum.append(result[i])
            else:
                
                daily_sum.append(result[i])
        daily_average.append(np.mean(daily_sum))
        all_daily_sum.append(daily_sum)
        
        return daily_average,all_daily_sum
    
    
    # a visualization function used by the function above (monthly wise)
    def dailyAverage(self,temp_data_pd,temp_data,result,chosen):
        daily_average,_=compute_daily_average(temp_data,result)
        plt.figure(figsize = (20,10))
        plt.title(f'Average daily sentiments of {chosen} for the month')
        font = {'size': 10}
        plt.rc('font', **font)
        plt.plot(temp_data_pd.date.unique(),daily_average)
        plt.xticks(rotation=20)
        plt.show()

    # a visualization function used by the function above (day wise)
    def daily_details(self,val,temp_data,result,chosen,chosen_date):
        
        daily_average,all_daily_sum=compute_daily_average(temp_data,result)
        plt.figure(figsize = (20,10))
        font = {'size': 10}
        plt.rc('font', **font)
        plt.title(f'Daily sentiments of {chosen} for day {chosen_date}')
        plt.plot(np.arange(1,len(all_daily_sum[val])+1),all_daily_sum[val])
        plt.xticks(rotation=20)
        plt.show()

    #call this function to get the buy and sell advice based on if the grade of the tweets retrieved is higher than the \n
    # mean of the tweets on that day, buy if so, sell otherwise.
    def getAdviceQuali(self):
        return 'buy' if self.avg_per_day[-1] >= np.asarray(self.avg_per_day).mean() else 'sell'


