

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

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    print(table)
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers


# save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=True):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2020, 4, 23)

    for ticker in tickers:
        ticker = ticker.replace(".", "-")

        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                print('stock_dfs/{}.csv'.format(ticker))
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except:
                print("Error finding " + ticker)
        else:
            print('Already have {}'.format(ticker))


# get_data_from_yahoo()

def get_metrics_data(metric="margin"):
    # with open("sp500tickers.pickle", "rb") as f:
    #     tickers = pickle.load(f)

    if not os.path.exists('stock_metrics'):
        os.makedirs('stock_metrics')
    ticker = "EAF"
    if not os.path.exists('stock_metrics/{}.csv'.format(ticker)):
        try:
            print('stock_metric/{}.csv'.format(ticker))
            r = requests.get(
                'https://finnhub.io/api/v1/stock/metric?symbol=' + ticker + '&metric=' + metric + '&token=' + key.FINNHUB + '&format=csv')

            print("Request received, reading json")
            print(r.json())
            # print(json_data)

            df = pd.read_json(r)
            print("json read, writing to csv")
            df.to_csv('stock_metric/{}.csv'.format(ticker))
            # print(r.json())
        except Exception as err:
            print("Error finding " + ticker + " error: " + err)
    else:
        print('Already have {}'.format(ticker))


# get_metrics_data()
def graph_candles(ticker):
    if os.path.exists('stock_dfs/{}.csv'.format(ticker.replace(".", "-"))):
        print('stock_dfs/{}.csv'.format(ticker.replace(".", "-")))
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker.replace(".", "-")))

    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)
    df['sma'] = df['Close'].rolling(window=10, min_periods=0).mean()

# graph_candles("AAPL")

def check_etf(ticker, start_date=today(), end_date=today(), cash=10000.00, resolution="D"):
    global currently_bought
    global current_close
    shares_held = 0
    purchase_price = 0.00
    buys = 0
    sells = 0
    starting_balance = cash
    periods = end_date - start_date
    periods = periods.days + 1
    start = start_date - pd.DateOffset(days=25)
    start = start.replace().timestamp()
    end = end_date.timestamp()
    # print(ticker, ":", start_date.date(), "to", end_date.date(), "(", periods, "days)")
    df = web.DataReader(ticker, 'yahoo', start_date, end_date)
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)
    df['sma'] = df['Close'].rolling(window=10, min_periods=0).mean()


    print(df.head(15))
    print(df['Close'])
    r = requests.get('https://finnhub.io/api/v1/indicator?symbol=' + ticker +
                     '&resolution=' + resolution + '&from=' + str(int(start)) + '&to=' + str(int(end)) +
                     '&indicator=sma&timeperiod=10&token=' + key.FINNHUB)
    json_r_sma = r.json()
    print(json_r_sma)
    print(ticker, ":", start_date.date(), "to", end_date.date(), "(", len(json_r_sma["c"])-25, " trading days)")



    r = requests.get('https://finnhub.io/api/v1/indicator?symbol=' + ticker +
                     '&resolution=' + resolution + '&from=' + str(int(start)) + '&to=' + str(int(end)) +
                     '&indicator=macd&fastperiod=10&slowperiod=17&signalperiod=9&token=' + key.FINNHUB)
    json_r_macd = r.json()
    # print(json_r_macd)
    r = requests.get('https://finnhub.io/api/v1/indicator?symbol=' + ticker +
                     '&resolution=' + resolution + '&from=' + str(int(start)) + '&to=' + str(int(end)) +
                     '&indicator=stoch&slowkperiod=14&slowdperiod=5&token=' + key.FINNHUB)
    json_r_stochastic = r.json()
    # df = pd.DataFrame(columns=['date','ticker','close','sma10','macdHist'])
    with open('sma.json','w') as f:
        json.dump(json_r_sma, f)
    df = pd.read_json(r'C:\Users\AC720\PycharmProjects\data_collector\sma.json')
    with open('macd.json','w') as f:
        json.dump(json_r_macd, f)
    df_macd = pd.read_json(r'C:\Users\AC720\PycharmProjects\data_collector\macd.json')
    df_macd['sma'] = df_macd['c'].rolling(window=10, min_periods=0).mean()
    print(df.tail(15))
    print(df_macd)
    # df_macd.head()
    # df_macd.plot()

    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    # candlestick2_ohlc(ax1, df['Open'],df['High'],df['Low'],df['Close'], width=2, colorup='g')
    # ax1.plot(df_macd.index, df_macd['c'])
    # ax1.plot(df_macd.index, df_macd['sma'])
    ax2.bar(df_macd.index, df_macd['macdHist'])
    plt.show()
    # print(json_r_stochastic)

    # r = requests.get('https://api.tradier.com/v1/markets/calendar',
    #                         params={'month': '02', 'year': '2020'},
    #                         headers={'Authorization': 'Bearer ak5bMMbkazkodUGDrLvGhnMDM9GB', 'Accept': 'application/json'})
    # json_r_calendar = r.json()
    # print(r.status_code)
    # print(json_r_calendar)
    month = start_date.month
    year = start_date.year
    day = start_date.day
    # for i in range(0,24):
    #     if

    for i in range(25, len(json_r_sma["c"])):
        # print("Loop:", i, "of", len(json_r_sma["c"]))
        if currently_bought:
            # print(i)

            if json_r_sma["c"][i] <= json_r_sma["sma"][i]:
                if json_r_macd["macdHist"][i] < 0:
                    if json_r_stochastic["slowk"][i] < 90:
                        tempdate = pd.to_datetime(start_date.date()) + pd.DateOffset(days=i)
                        # print("Sell", tempdate.date(), ":", shares_held, "@ $", json_r_sma["c"][i], "+ $", round(cash,2),
                        #       ", total portfolio $", round(shares_held * json_r_sma["c"][i] + cash))
                        cash = shares_held * json_r_sma["c"][i] + cash
                        shares_held = 0
                        currently_bought = False
                        sells += 1

        else:
            # print(i)
            if json_r_sma["c"][i] > json_r_sma["sma"][i]:
                if json_r_macd["macdHist"][i] > 0:
                    # print(json_r_stochastic["slowk"][1])
                    if json_r_stochastic["slowk"][i] > 10:
                        shares_held = math.floor(cash / json_r_sma["c"][i])
                        cash = cash - (shares_held * json_r_sma["c"][i])
                        currently_bought = True
                        tempdate = pd.to_datetime(start_date.date()) + pd.DateOffset(days=i)
                        # print("Buy ", tempdate.date(), ":", shares_held, "@ $", json_r_sma["c"][i], "+ $", round(cash,2),
                        #       ", total portfolio $", round(shares_held * json_r_sma["c"][i] + cash))
                        buys += 1

    final_balance = shares_held * json_r_sma["c"][-1] + round(cash)
    print(ticker, shares_held, "@ $", json_r_sma["c"][-1], "+ $", round(cash), ", total portfolio $", round(final_balance))
    print("$", starting_balance, "-> $", round(final_balance), "=", round((final_balance-starting_balance)/starting_balance*100), "% in", periods, "days", "ARORC", round((final_balance-starting_balance)/starting_balance*100/periods*365), "%", buys, "buys,", sells, "sells")

    # print(buys, "buys,", sells, "sells")
    print("1st Day close: $", json_r_sma["c"][25], "Last Day Close: $", json_r_sma["c"][-1],"Buy and hold would have been ", round((json_r_sma["c"][-1]-json_r_sma["c"][25])/json_r_sma["c"][25]*100,1), "% or ARORC",
          round((json_r_sma["c"][-1]-json_r_sma["c"][25])/json_r_sma["c"][25]*100/periods*365,1),"%")
    if ((json_r_sma["c"][-1]-json_r_sma["c"][25])/json_r_sma["c"][25]) > ((final_balance-starting_balance)/starting_balance):
        print("Hold is better by    ", round((((json_r_sma["c"][-1]-json_r_sma["c"][25])/json_r_sma["c"][25]) - ((final_balance-starting_balance)/starting_balance))*100,1),"% or ARORC",round(((((json_r_sma["c"][-1]-json_r_sma["c"][25])/json_r_sma["c"][25]) - ((final_balance-starting_balance)/starting_balance))) * 100 / periods * 365, 1),"%")
    else:
        print("Buy/Sell is better by", round(((((final_balance - starting_balance) / starting_balance)) - ((json_r_sma["c"][-1] - json_r_sma["c"][25]) / json_r_sma["c"][25])) * 100, 1), "% or ARORC", round(((((final_balance - starting_balance) / starting_balance)) - ((json_r_sma["c"][-1] - json_r_sma["c"][25]) / json_r_sma["c"][25])) * 100 / periods * 365, 1),"%")

    df['sma'].plot()

start_date = dt.datetime(2015,1,1)
end_date = today()

check_etf("VGT",start_date,end_date)
# check_etf("VUG",start_date,end_date)
# check_etf("RSP",start_date,end_date)
# check_etf("VGT",start_date,end_date)
# check_etf("VV",start_date,end_date)
# check_etf("MGV",start_date,end_date)
# check_etf("SPY",start_date,end_date)


def check_calendar(start_date):
    date = start_date
    while date <= today():
        # print(date)
        try:
            r = requests.get('https://api.tradier.com/v1/markets/calendar',
                             params={'month': date.month, 'year': date.year},
                             headers={'Authorization': 'Bearer ak5bMMbkazkodUGDrLvGhnMDM9GB', 'Accept': 'application/json'})
            # return r.json()
            json_r_calendar = r.json()
            # print(r.status_code)
            # print(json_r_calendar)
            print(json_r_calendar["calendar"]["days"]["day"][start_date.day-1]["date"],json_r_calendar["calendar"]["days"]["day"][start_date.day-1]["status"])
        except:
            print("Error pulling data for", date.year, "-", date.month)
        date += relativedelta(years=+1)

# check_calendar(start_date)

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        print(ticker)
        if os.path.exists('stock_dfs/{}.csv'.format(ticker.replace(".", "-"))):
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker.replace(".", "-")))
            df.set_index('Date', inplace=True)

            print(ticker)
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


# compile_data()

def visualise_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    # df['AAPL'].plot()
    # plt.show()
    df_corr = df.corr()

    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

# visualise_data()
