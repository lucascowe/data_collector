import bs4 as bs
from dateutil.utils import today
import json
from dateutil.relativedelta import relativedelta
# import httplib
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick2_ohlc
import matplotlib.dates as mdates
import math
import numpy as np
import requests
import pandas as pd
import pandas_datareader as web
import pickle
import os
import datetime as dt

import key
style.use('ggplot')

currently_bought = False
lastday_close = 0.00
current_close = 0.00
cash = 10000.00

def save_us_etf_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds')
    # resp = requests.get('https://en.wikipedia.org/wiki/List_of_Australian_exchange-traded_funds')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    lines = soup.find_all('li')
    tickers = []

    i = 1
    for line in lines:
        if line.text.find('|') > 0:
            temp = line.text.split('|')
            # print("Line", i, len(line.text), temp[1][:temp[1].find(')')], " - ", line.text)
            # print(len(temp),"text segments")

            for j in range(0,len(temp)):
                # print("In for loop - Line ", j, "Actual segment:", temp[j],temp[j].find(')'))
                # print(temp[j][0:5])
                # if temp[j].find(')') > 0:
                if temp[j][0:5].find(')') > 0:
                    # print("Segment", j, temp[j][0:temp[j].find(')')])
                    # print(temp[i])
                    tickers.append(temp[j][0:temp[j].find(')')])

        i += 1

    # table = soup.find('table', {'class': 'wikitable sortable'})
    #
    # for row in table.findAll('tr')[1:]:
    #     ticker = row.findAll('td')[0].text
    #     ticker = ticker[:-1]
    #     tickers.append(ticker)

    with open("us_etfs.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers

# print(save_us_etf_tickers())

def get_data_from_yahoo(reload_etf=True):
    if reload_etf:
        tickers = save_us_etf_tickers()
    else:
        with open("us_etfs.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('us_etf_dfs'):
        os.makedirs('us_etf_dfs')

    start = dt.datetime(2000, 1, 1)
    end = today()

    for ticker in tickers:
        ticker = ticker.replace(".", "-")

        if not os.path.exists('us_etf_dfs/{}.csv'.format(ticker)):
            try:
                print('us_etf_dfs/{}.csv'.format(ticker))
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('us_etf_dfs/{}.csv'.format(ticker))
            except:
                print("Error finding " + ticker)
        else:
            print('Already have {}'.format(ticker))

start_date = dt.datetime(2015,1,1)
# get_data_from_yahoo()

def stochastics( dataframe, low='Low', high='High', close='Close', k=14, d=3 ):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal
    When the %K crosses below %D, sell signal
    """

    df = dataframe.copy()

    # Set minimum low and maximum high of the k stoch
    low_min  = df[low].rolling( window = k ).min()
    high_max = df[high].rolling( window = k ).max()

    # Fast Stochastic
    df['k_fast'] = 100 * (df[close] - low_min)/(high_max - low_min)
    df['d_fast'] = df['k_fast'].rolling(window = d).mean()

    # Slow Stochastic
    df['Slow_K'] = df["d_fast"]
    df['d_slow'] = df['Slow_K'].rolling(window = d).mean()
    df.drop(['k_fast','d_slow','d_fast'], 1, inplace=True)

    return df

def computeRSI(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval)/n
        down = (down * (n - 1) + downval)/n

        rs = up / down
        rsi[i] = 100. / (1 + rs)

    return rsi



def check_etfs(ticker, start_date, end_date=today()):
    with open("us_etfs.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame
    df = pd.read_csv('us_etf_dfs/{}.csv'.format(ticker))
    df.set_index('Date', inplace=True)
    df['%Volatil'] = round((df['High']-df['Close'])/df['Open']*100,2)
    # df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1, inplace=True)
    df['SMA'] = df['Close'].rolling(window=10, min_periods=0).mean()
    # df['RSI'] = computeRSI(df['Close'])
    # macd = computeMACD(df['Close'])

    df['EMA17'] = df.iloc[:, 3].ewm(span=17, adjust=False).mean()
    df['EMA8'] = df.iloc[:, 3].ewm(span=8, adjust=False).mean()
    # df['emaSlow17'] = ExpMovingAverage(df['Close'],17)
    # df['emaFast8'] = ExpMovingAverage(df['Close'], 8)
    df['MACD'] = df['EMA8'] - df['EMA17']
    df['Signal9'] = df.iloc[:, 10].ewm(span=9, adjust=False).mean()
    # df['Signal9'] = ExpMovingAverage(/macd, 9)
    df['MACDHist'] = df['MACD']- df['Signal9']
    df = stochastics(df)
    df.drop(['High','Low', 'Open','EMA17', 'EMA8', 'MACD', 'Signal9'], 1, inplace=True)
    df.loc[df['Close'] > df['SMA'], 'SMA_Status'] = "G"
    df.loc[df['Close'] <= df['SMA'], 'SMA_Status'] = "R"
    df.loc[df['MACDHist'] > 0, 'MACD_Status'] = "G"
    df.loc[df['MACDHist'] <= 0, 'MACD_Status'] = "R"
    df.loc[df['Slow_K'] >= 80, 'Stochast_Status'] = "G"
    df.loc[df['Slow_K'] <= 20, 'Stochast_Status'] = "R"
    df.loc[(df['SMA_Status'] == "G") & (df['MACD_Status'] == "G") & (df['Slow_K'] > 20), 'Stochast_Status'] = "G"
    df.loc[(df['SMA_Status'] == "R") & (df['MACD_Status'] == "R") & (df['Slow_K'] < 80), 'Stochast_Status'] = "R"
        # df['Status'] = "Buy"
    # for i in range(17, len(df["Close"])):
    #     if df['Close'][i] > df['SMA'][i]:
    #         if df['MACDHist'][i] > 0:
    #             if df['Slow_K'][i] > 20:
    #                 df['Status'][i] = 'Bought'



    pd.set_option('max_columns', 30)
    print(df.tail(20))

# check_etfs("VUG", start_date)
check_etfs("VGT", start_date)

def compile_data():
    with open("aus_etfs.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('etf_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename()