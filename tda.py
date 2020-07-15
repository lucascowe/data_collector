import json

import requests
from key import TDA

header = {"Authorization": f"Bearer {TDA}"}

symbol = 'voo'

url = f"https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory"

payload = {
    "periodType": "ytd",
}

response = requests.get(url, headers=header, json=payload)

print(f"Response:\n{json.dumps(response.json(), indent=2)}")
