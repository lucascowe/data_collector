import json
import time
import datetime
import urllib.request

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader as web
import pickle
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from matplotlib import style

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import key

ticker_filename = "smp500tickers.pickle"
stock_dir = "../stocks_prices"
smp_joined_closes = "smp_joined_closes.csv"
data_folder = "data"
msn_urls_folder = "company_info"


def save_day(df):
    print("saving")


def get_company_info(ticker):
    r = requests.get('https://finnhub.io/api/v1/stock/profile2?symbol=' + ticker + '&token=' + key.FINNHUB)
    company_info = open(os.path.join(data_folder, msn_urls_folder, ticker + ".json"), 'r').read()
    company_info = json.loads(company_info)
    retrieved_info = r.json()
    for value in retrieved_info.keys():
        company_info[value] = retrieved_info[value]
    print(json.dumps(company_info, indent=2, sort_keys=True))
    open(os.path.join(data_folder, msn_urls_folder, ticker + ".json"), 'w')\
        .write(json.dumps(company_info, indent=2, sort_keys=True))
    return company_info

# /stock/financials-reported?symbol=AAPL
#
# /stock/financials-reported?cik=320193&freq=quarterly
#
# /stock/financials-reported?accessNumber=0000320193-20-000052

def get_financials_reported(ticker, period="annual", accessNumber=None):
    """
    :param accessNumber: report access number
    :param ticker:
    :param period: annual or quarterly
    :return:
    """
    r = requests.get('https://finnhub.io/api/v1/stock/financials-reported?symbol=' + ticker +
                     '&freq=' + period + '&token=' + key.FINNHUB)
    company_info = open(os.path.join(data_folder, msn_urls_folder, ticker + ".json"), 'r').read()
    company_info = json.loads(company_info)
    retrieved_info = r.json()
    for value in retrieved_info.keys():
        company_info[value] = retrieved_info[value]
    # print(json.dumps(company_info, indent=2, sort_keys=True))
    open(os.path.join(data_folder, msn_urls_folder, ticker + ".json"), 'w') \
        .write(json.dumps(company_info, indent=2, sort_keys=True))
    return company_info



def load_financials(ticker):
    if not os.path.isfile(os.path.join(data_folder, msn_urls_folder, ticker + ".json")):
        get_company_info(ticker)
        company_info = get_financials_reported(ticker)
    else:
        company_info = open(os.path.join(data_folder, msn_urls_folder, ticker + ".json"), 'r').read()
        company_info = json.loads(company_info)
    return company_info

report_list = []
# for tic in os.listdir(os.path.join("data", "company_info")):
#     ticker_name = tic.split('.')[0]
#     print(f"Checking {ticker_name}")
#     try:
#         result = get_financials_reported(ticker_name)
#         if "data" in result and result["data"]:
#             report_list.append(ticker_name)
#             print(f"Found reports")
#     except Exception as e:
#         print(f"error getting tic: {e}")
# open(os.path.join("data", "company_info", "1_companies_with_reports.json")).write(json.dumps({"companies": report_list}))
# print(f'Financials reported: \n{json.dumps(get_financials_reported("amzn"), indent=2, sort_keys=True)}')


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text
        ticker = str(ticker).replace(".", "-")
        tickers.append(ticker[:-1])
    with open("smp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


def get_html_val(line):
    return (str(line).split('>')[1]).split('<')[0]


def get_day_prices_bloomberg(ticker):
    try:
        user_agent = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_4; en-US) AppleWebKit/534.3 (KHTML, like Gecko) Chrome/6.0.472.63 Safari/534.3'
        headers = {'User-Agent': user_agent}
        bloomberg_url = f"https://www.bloomberg.com/markets/api/bulk-time-series/price/{ticker}%3AUS?timeFrame=1_DAY"
        req = urllib.request.Request(bloomberg_url, None, {'User-agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'})
        page = urllib.request.urlopen(req)
        data = json.loads(page.read().decode('ascii'))
        # print(f"{json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"Error getting bloomberg prices: {e}")
    return data


def get_top_movers_yahoo():
    tickers = []
    try:
        yahoo_movers_address = "https://finance.yahoo.com/most-active/"
        resp = requests.get(yahoo_movers_address, timeout=10)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table')
        # print(f"Table: {table}")
        for row in table.find_all('tr')[1:]:
            ticker = row.find_all('td')[0].text
            # print(f"row: {ticker}")
            ticker = str(ticker).replace(".", "-")
            if ticker != "":
                tickers.append(ticker[:-1])
        # with open("smp500tickers.pickle", "wb") as f:
        #     pickle.dump(tickers, f)
    except Exception as e:
        print(f"ERROR: get_top_movers_yahoo: {e}")
    return tickers


# print(f"Top movers")
# top_movers = get_top_movers_yahoo()
# for ticker in top_movers:
#     get_day_prices_bloomberg(ticker)


def get_msn_url_for_ticker(ticker, force_reload=False):
    ticker = str(ticker)
    msn_address = "https://www.msn.com/en-us/money"
    timeout = 6
    ticker_path = os.path.join(data_folder, msn_urls_folder, ticker + ".json")
    if not force_reload:
        if os.path.isfile(ticker_path):
            ticker_info = open(ticker_path, 'r').read()
            ticker_info = json.loads(ticker_info)
            if "msn_url" in ticker_info:
                if ticker_info["msn_url"] != msn_address:
                    print(f"{ticker} -> {ticker_info['msn_url']}")
                    return ticker_info["msn_url"]
    chrome_driver = os.path.join("chromedriver")
    # print(f"location is {chrome_driver}")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=chrome_options)
    driver.get(msn_address)
    driver.find_element_by_id("finance-autosuggest").send_keys(ticker)
    driver.find_element_by_id("finance-autosuggest").send_keys(u'\ue007')
    print(f"Searching for {ticker} msn money url")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if driver.current_url != msn_address:
            print(f"url found in {time.time() - start_time}")
            break
    url = driver.current_url
    driver.close()
    if url != msn_address:
        if os.path.isfile(ticker_path):
            ticker_info = open(ticker_path, 'r').read()
            ticker_info = json.loads(ticker_info)
            ticker_info["msn_url"] = url
        else:
            ticker_info = {"msn_url": url}
        with open(ticker_path, 'w') as save_ticker:
            save_ticker.write(json.dumps(ticker_info))
        print(f"ticker url {url} saved to {ticker_path}")
        return url
    else:
        print(f"Failed to get ticker url for {ticker} after {time.time() - start_time} seconds")
        return None


def get_current_price_msn(ticker):
    web_address = get_msn_url_for_ticker(ticker)
    print(f"Scraping {web_address} at {dt.datetime.now().hour}:{dt.datetime.now().minute}")
    resp = requests.get(web_address)
    soup = bs.BeautifulSoup(resp.text, 'html.parser')
    prices = soup.find('body')
    div = prices.find('div', {'class': 'col2 quotedata-livequote'})
    # need to add , removal
    cur_price = get_html_val(div.find('span', {'class': 'currentval'}))
    if ',' in cur_price:
        tmp = ""
        for letter in cur_price:
            if letter.isnumeric() or letter =='.':
                tmp += letter
        cur_price = float(tmp)
    else:
        cur_price = float(cur_price)
    print(f"{ticker} price {cur_price}")
    return cur_price


def get_ticker_price(ticker, start, end, source='yahoo', save=True):
    df = web.DataReader(ticker, source, start, end)
    if save:
        filename = os.path.join(stock_dir, ticker + '.csv')
        df.to_csv(filename)
    return df


def get_500_from_yahoo(reload_smp500=False, reload_data=True):
    if reload_smp500:
        tickers = save_sp500_tickers()
    else:
        with open("smp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('../stocks_prices'):
        os.mkdir(stock_dir)
    start = dt.date(2000, 1, 1)
    end = dt.date(2020, 4, 23)
    print(f'Downloading smp500 {dt.datetime}')
    i = 0
    missed = []
    for ticker in tickers:
        i += 1
        ticker = str(ticker).replace(".", "-")
        filename = os.path.join(stock_dir, ticker + '.csv')
        print(f"{i}. Downloading {ticker}")
        try:
            if not os.path.exists(filename):
                get_ticker_price(ticker, start, end)
            else:
                print("You already have it")
        except:
            print(f'Missed it')
            missed.append(ticker)
    with open("../missed.pickle", "wb") as f:
        pickle.dump(missed, f)
    print(f"missed tickers : {missed}")


def compile_data():
    with open(ticker_filename, 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):

        df = pd.read_csv(os.path.join(stock_dir, ticker + ".csv"))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv(smp_joined_closes)


price_folder = os.path.join("data", "prices")


style.use('ggplot')


# get_500_from_yahoo(reload_smp500=True)

# compile_data()
# smp = get

def visualize_data():
    df = pd.read_csv(smp_joined_closes)
    # df['AAPL'].plot()
    # plt.show()
    df_corr = df.corr()

    df_corr.plot()
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


def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv(smp_joined_closes, index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.025
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_feature_set(ticker):
    tickers, df = process_data_for_labels(ticker)

    df[ticker + "_target"] = list(map(buy_sell_hold,
                                      df[ticker + '_1d'],
                                      df[ticker + '_2d'],
                                      df[ticker + '_3d'],
                                      df[ticker + '_4d'],
                                      df[ticker + '_5d'],
                                      df[ticker + '_7d'],
                                      df[ticker + '_7d']
                                      ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df[ticker + "_target"].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_feature_set(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf_rfor = RandomForestClassifier(n_estimators=50, random_state=1)

    # clf = VotingClassifier(estimators=[
    #                                    ('lsvc', svm.LinearSVC()),
    #                                    ('knn', neighbors.KNeighborsClassifier())
    #                                    ('rfor', clf_rfor)
    #                                   ], voting='hard')

    clf_rfor.fit(X_train, y_train)

    confidency = clf_rfor.score(X_test, y_test)
    print(f"Accuracy: {confidency}")
    predictions = clf_rfor.predict(X_test)

    print(f"Predicted spread: {Counter(predictions)}")

    return confidency


# tickers, data_f = process_data_for_labels('XOM')
# print(data_f)
# return 0
# do_ml("BAC")
# visualize_data()
# get_current_price_msn('TSLA')
# get_current_price_msn('LULU')
# get_current_price_msn('MSFT')
# get_current_price_msn('VOO')
# get_current_price_msn('VUG')
# get_current_price_msn('MFA')

# start = dt.datetime(2010, 7, 1)
# end = dt.datetime(2020, 4, 27)

# print(get_ticker_price('tsla', start, end))

# df = web.DataReader('TSLA', 'yahoo', start, end)
# df.to_csv('tsla.cvs')
# print(df.head(600))

# df = pd.read_csv('tsla.cvs', parse_dates=True, index_col=0)
# print(df)

# df['Adj Close'].plot()
# plt.show()

# df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean().dropna()

# print(df)

# ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)

# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])
#
# plt.show()

# Resampling
# df_ohlc = df['Adj Close'].resample('10D').ohlc()
# df_vol = df['Volume'].resample('10D').sum()
#
# df_ohlc.reset_index(inplace=True)
#
# df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
#
# ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)
# ax1.xaxis_date()

# candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
# ax2.fill_between(df_vol.index.map(mdates.date2num), df_vol.values, 0)
# plt.show()

# print(df_ohlc)

# print(f"SMP500 tickers: {save_sp500_tickers()}")
