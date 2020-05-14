import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader as web
import pickle
import requests

names = []
prices = []
changes = []
percentChanges = []
marketCaps = []
totalVolumes = []
circulatingSupplys = []

# for i in range(0, 11):
active_stocks = "https://finance.yahoo.com/most-active/?offset=25&count=25"
active_yahoo = "https://finance.yahoo.com/most-active/"
ticker_yahoo = "https://finance.yahoo.com/quote/TSLA?p=TSLA&.tsrc=fin-srch"
    # active_stocks = "https://in.finance.yahoo.com/most-active?offset=" + str(
    #     i) + "&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;count=100"
r = requests.get(ticker_yahoo)
soup = bs.BeautifulSoup(r.text, "lxml")

body = soup.find('body')
layer = body.find('div', attrs={'id': 'app'})
layer = layer.find('div', attrs={'id': 'app'})
layer = layer.find('div', attrs={'id': 'app'})
layer = layer.find('div', attrs={'id': 'app'})
print(body)
# return 0
# print(f"Read {r.text[:300]}")
# for listing in soup.find_all('tr'):
# # for listing in soup.find_all('tr', attrs={'class': 'SimpleDataTableRow'}):
#     print(f"listing: {listing}")
#     for name in listing.find_all('td', attrs={'aria-label': 'Name'}):
#         print(f"name: {name}")
#         names.append(name.text)
#     for price in listing.find_all('td', attrs={'aria-label': 'Price (intraday)'}):
#         print(f"price: {price}")
#         prices.append(price.find('span').text)
#     for change in listing.find_all('td', attrs={'aria-label': 'Change'}):
#         print(f"change: {change}")
#         changes.append(change.text)
#     for percentChange in listing.find_all('td', attrs={'aria-label': '% change'}):
#         print(f"percentChange: {percentChange}")
#         percentChanges.append(percentChange.text)
#     for marketCap in listing.find_all('td', attrs={'aria-label': 'Market cap'}):
#         print(f"marketCap: {marketCap}")
#         marketCaps.append(marketCap.text)
#     for totalVolume in listing.find_all('td', attrs={'aria-label': 'Avg vol (3-month)'}):
#         print(f"totalVolume: {totalVolume}")
#         totalVolumes.append(totalVolume.text)
#     for circulatingSupply in listing.find_all('td', attrs={'aria-label': 'Volume'}):
#         print(f"circulatingSupply: {circulatingSupply}")
#         circulatingSupplys.append(circulatingSupply.text)
#
# pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges, "Market Cap": marketCaps,
#               "Average Volume": totalVolumes, "Volume": circulatingSupplys})

