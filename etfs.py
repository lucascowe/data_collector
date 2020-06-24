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

    start = dt.datetime(2020, 1, 1)
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

def calc_days(start_date, end_date):
    """
    :param start_date: date as string in yyyy-MM-dd format
    :param end_date: date as string in yyyy-MM-dd format
    :return: number of days between the two dates
    """
    date_range = dt.datetime(int(end_date[0:4]), int(end_date[5:-3]), int(end_date[8:])) - \
                 dt.datetime(int(start_date[0:4]), int(start_date[5:-3]), int(start_date[8:]))
    if date_range.days >= 360:
        return 365
    else:
        return date_range.days

def save_data(ticker, df, location="etf_arrow_working"):
    try:
        if not os.path.isdir(location):
            os.mkdir(location)
        filename = f"{ticker}_arrow.csv"
        file_path = os.path.join(location, filename)
        if os.path.isfile(file_path):
            print(f"File existing, saving without header")
            df.to_csv(file_path, mode='a', header=False)
        else:
            print(f"File not found, creating new")
            df.to_csv(file_path)
    except Exception as e:
        print(f"Error saving prices to file for {file_path}: {e}")


def check_etf(ticker, start_date, end_date=today()):

    main_df = pd.DataFrame
    df = pd.read_csv('us_etf_dfs/{}.csv'.format(ticker))

    df['%Volatil'] = round((df['High']-df['Close'])/df['Open']*100,2)
    # df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], 1, inplace=True)
    df['SMA'] = df['Close'].rolling(window=10, min_periods=0).mean()
    df['EMA17'] = df['Close'].ewm(span=17, adjust=False).mean()
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['MACD'] = df['EMA8'] - df['EMA17']
    df['Signal9'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDHist'] = df['MACD'] - df['Signal9']
    df = stochastics(df)
    # df.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close', 'EMA17', 'EMA8', 'MACD', 'Signal9', '%Volatil'], 1, inplace=True)
    perform_df = pd.DataFrame(columns=['Year', '1st_Open', 'Fin_Close', 'Shares', 'Cash', 'Tot_Val', 'T_ARORC', 'H_ARORC', 'Buys', 'Sells'])
    perform_df.set_index('Year', inplace=True)
    start_index = 0
    starting_cash = 10000.0
    currently_bought = False
    col_names = ['Date','Action','Total_Val','Shares','Cash','Close','SMA','MACDHist','SlowK']
    buys_sells_df = pd.DataFrame(columns=col_names)
    buys_sells_df.set_index('Date', inplace=True)
    for ind in df.index:
        if dt.datetime(int(df.loc[ind, "Date"][0:4]), int(df.loc[ind, "Date"][5:-3]), int(df.loc[ind, "Date"][8:])) >= start_date:
            if start_index == 0:
                start_index = ind
                year = int(df.loc[ind, 'Date'][0:4])
                perform_df.loc[year, '1st_Open'] = df.loc[ind, "Open"]
                perform_df.loc[year, 'Cash'] = starting_cash

            elif year != int(df.loc[ind, 'Date'][0:4]):
                perform_df.loc[year, 'Fin_Close'] = df.loc[ind - 1, "Close"]
                date_range = calc_days(df.loc[start_index, "Date"],df.loc[ind-1, "Date"])

                perform_df.loc[year, 'H_ARORC'] = round(((perform_df.loc[year, 'Fin_Close'] - perform_df.loc[year, '1st_Open'])
                                                         / perform_df.loc[year, '1st_Open'] * (date_range / 365)) * 100, 1)
                perform_df.loc[year, 'T_ARORC'] = round((perform_df.loc[year, 'Tot_Val'] - starting_cash) / starting_cash *
                                                        (date_range / 365) * 100, 1)

                # Set new year
                starting_cash = perform_df.loc[year, 'Tot_Val']
                year = int(df.loc[ind, 'Date'][0:4])
                perform_df.loc[year, '1st_Open'] = df.loc[ind - 1, "Close"]
                perform_df.loc[year, 'Shares'] = perform_df.loc[year - 1, 'Shares']
                perform_df.loc[year, 'Cash'] = perform_df.loc[year - 1, 'Cash']
                perform_df.loc[year, 'Tot_Val'] = perform_df.loc[year - 1, 'Tot_Val']
                perform_df.loc[year, 'Buys'] = 0
                perform_df.loc[year, 'Sells'] = 0
                start_index = ind

            if not currently_bought:
                if df.loc[ind, "SMA"] < df.loc[ind, "Close"]:
                    if df.loc[ind, "MACDHist"] > 0:
                        if df.loc[ind, "Slow_K"] > 20:
                            if df.loc[ind - 1, "Slow_K"] < df.loc[ind, "Slow_K"]:
                                perform_df.loc[year, 'Shares'] = math.floor(perform_df.loc[year, 'Cash'] / df.loc[ind, "Close"])
                                perform_df.loc[year, 'Cash'] = perform_df.loc[year, 'Cash'] - perform_df.loc[year, 'Shares'] \
                                                               * df.loc[ind, "Close"]
                                perform_df.loc[year, 'Tot_Val'] = perform_df.loc[year, 'Shares'] * df.loc[ind, "Close"] \
                                                                  + perform_df.loc[year, 'Cash']
                                perform_df.loc[year, 'Buys'] += 1
                                currently_bought = True
                                # print("Shares:", shares, "@", round(df.loc[ind, "Close"], 2), " + $", round(cash, 2),
                                buys_sells_df.loc[df.loc[ind, "Date"]] = 'Buy', perform_df.iloc[-1]['Tot_Val'], \
                                            perform_df.loc[year, 'Shares'], perform_df.loc[year, 'Cash'], df.loc[ind, 'Close'], \
                                            df.loc[ind, 'SMA'], df.loc[ind, 'MACDHist'], df.loc[ind, 'Slow_K']
                                #       "Total Value $", round(total_val, 2))

            else:
                if df.loc[ind, "SMA"] >= df.loc[ind, "Close"]:
                    if df.loc[ind, "MACDHist"] <= 0:
                        if df.loc[ind, "Slow_K"] < 80:
                            if df.loc[ind - 1, "Slow_K"] > df.loc[ind, "Slow_K"]:
                                perform_df.loc[year, 'Cash'] = perform_df.loc[year, 'Cash'] + perform_df.loc[year, 'Shares'] * df.loc[ind, "Close"]
                                perform_df.loc[year, 'Shares'] = 0
                                perform_df.loc[year, 'Tot_Val'] = perform_df.loc[year, 'Cash']
                                perform_df.loc[year, 'Sells'] += 1
                                currently_bought = False
                                # print("Shares:", shares, "@", round(df.loc[ind, "Close"], 2), " + $", round(cash, 2),
                                #       "Total Value $", round(total_val, 2))
                                buys_sells_df.loc[df.loc[ind, "Date"]] = 'Sell', perform_df.iloc[-1]['Tot_Val'], \
                                                                         perform_df.loc[year, 'Shares'], perform_df.loc[
                                                                             year, 'Cash'], df.loc[ind, 'Close'], \
                                                                         df.loc[ind, 'SMA'], df.loc[ind, 'MACDHist'], \
                                                                         df.loc[ind, 'Slow_K']

    perform_df.loc[year, 'Fin_Close'] = df.loc[ind, "Close"]
    date_range = calc_days(df.loc[start_index, "Date"], df.loc[ind, "Date"])

    perform_df.loc[year, 'H_ARORC'] = round(((perform_df.loc[year, 'Fin_Close'] - perform_df.loc[year, '1st_Open'])
                                             / perform_df.loc[year, '1st_Open'] * (date_range / 365)) * 100, 1)
    perform_df.loc[year, 'T_ARORC'] = round((perform_df.loc[year, 'Tot_Val'] - starting_cash) / starting_cash *
                                            (date_range / 365) * 100, 1)

    perform_df.fillna(0, inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(perform_df)
    hold_final = round(math.floor(starting_cash / df.iloc[0]["Close"]) * df.iloc[-1]["Close"] +
          starting_cash - (math.floor(starting_cash / df.iloc[0]["Close"]) * df.iloc[0]["Close"]),2)

    date_range = calc_days(df.iloc[0]['Date'], df.iloc[-1]['Date'])
    if hold_final > perform_df.iloc[-1]['Tot_Val']:
        best_method = "hold"
    else:
        best_method = "Arrows"
    hold_arorc = round((hold_final - starting_cash) / starting_cash * (date_range / 365) * 100)
    arrows_arorc = round((perform_df.iloc[-1]['Tot_Val'] - starting_cash) / starting_cash * (date_range / 365) * 100)
    save_data(ticker, buys_sells_df)

    print(perform_df.iloc[0]['1st_Open'])
    print(perform_df.iloc[-1]['Fin_Close'])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(buys_sells_df)
    print(perform_df['Buys'].sum())
    return best_method, start_date.date(), df.loc[start_index, "Close"],df.loc[ind, "Date"], df.loc[ind, "Close"],\
           hold_final, starting_cash,hold_arorc, perform_df.iloc[-1]['Tot_Val'], round(arrows_arorc,2), \
           perform_df['Buys'].sum(), perform_df['Sells'].sum()

start_date = dt.datetime(2019,1,1)

print(check_etf("SPY", start_date))
# check_etf("VOO", start_date)
# check_etf("VGT", start_date)
# check_etf("SPY", start_date)
# check_etf("RSP",start_date)
# check_etf("VV",start_date)
# check_etf("MGV",start_date)
# check_etf("VAP", start_date)

def compile_data(test = False):
    with open("us_etfs.pickle","rb") as f:
        tickers = pickle.load(f)

    # main_df = pd.DataFrame

    col_names = ['Ticker','Best_Method','Start','Str_Close','Finish','Fin_Close','Start_$', 'Hold_$','Hold_ARORC', 'Arrows_$', 'Arrows ARORC', 'Buys', 'Sells']

    summary_df = pd.DataFrame(columns=col_names)
    summary_df.set_index('Ticker', inplace=True)
    i = 0
    for count, ticker in enumerate(tickers):
        try:
            summary_df.loc[ticker] = check_etf(ticker, start_date)
            i += 1
            if test and i == 5:
                return "Done"


        except:
            print("Error on ", ticker)

        # df = pd.read_csv('etf_dfs/{}.csv'.format(ticker))
        # df.set_index('Date', inplace=True)
        # df.rename()



    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(summary_df)


# compile_data(True)