# MAS
multi agent crypto currency trading platform

Description: This is a school project that the team had build to simulate an online trading platform for crypto currencies. 

The system involves suggestion agents such as MLP agents that computes and predicts the closing price of a crypto currencies based on its historical price (72hours). MA(moving average) agent that predict the trend of the market by calculating moving average etc.

These suggestion agents agent then pass the decision to the decider agent and CEO agents for deciding on the final action. And eventually the broker agent would carry out the said action and perform trading online, and pass back its profit and loss statement back to the ceo and decider agent for analysis and improves its decision making based on case based reasoning.

Instruction on how to run the program:

1. install all the required package through the requirements.txt by running pip install -r requirements.txt
2. python main.py
3. a link would pop up, click on the link and the user would be taken to the online trading platform written in python flask.
![Screenshot (387)](https://user-images.githubusercontent.com/65244703/166096267-7b6a9832-d793-4193-a1ed-68e4eec00373.png)
4. click on control panel, set the desired options and click run
5. wait for around 3 mins as the MLP and fuzzy logic is running in the background for suggestions
6. once done, go to the home page to view the powerbi report.
![Screenshot (389)](https://user-images.githubusercontent.com/65244703/166096315-ebb05455-96ca-45c0-9d5b-f73577ff9fe6.png)

Miscellaneous

The inner PowerBi report is constructed as follow:
![Screenshot (391)](https://user-images.githubusercontent.com/65244703/166096421-fdd04048-66fb-490d-a2d7-9c1c5a6f479b.png)

