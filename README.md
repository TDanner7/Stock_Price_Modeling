# Stock_Price_Modeling

## Imports

import pandas as pd


import numpy as np


from pathlib import Path


import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')

import yfinance as yahooFinance


from datetime import datetime, timedelta


from pandas.tseries.offsets import DateOffset


from finta import TA

from sklearn.preprocessing import StandardScaler


(from sklearn.pipeline import Pipeline)


from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestRegressor


from sklearn.ensemble import GradientBoostingRegressor


from sklearn.linear_model import LinearRegression


from sklearn.linear_model import Ridge

from sklearn.ensemble import StackingRegressor

from sklearn.ensemble import VotingRegressor

import xgboost as xgb


from xgboost.sklearn import XGBRegressor

import quantstats as qs

import warnings


warnings.filterwarnings("ignore")

## Random_Forest

###

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

## Gradient_Boost

###

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

## XGradient_Boost

###

https://xgboost.readthedocs.io/en/latest/install.html

## Linear_Regression

###

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

## Ridge

###

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

## Stacking_Models

###

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html

## Voting_Classifier

###

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html

