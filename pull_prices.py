import urllib.request
import time
ticker = 'TSLA'

def pull_data(stock):
    try:
        endpoint = f'http://chartapi.finance.yahoo.com/instrument/1.0/{stock}/chartdata;type=quote;range=1y/csv'
        source = urllib.request.urlopen(endpoint)
        print(f"Source:\n{source}")
    except Exception as e:
        print(f"Exception: {e}")

pull_data(ticker)