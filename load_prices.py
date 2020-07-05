import pandas as pd
import os

for f in os.listdir(os.path.join("data", "stock_prices"))[:50]:
    ticker_prices = pd.read_csv(f)


