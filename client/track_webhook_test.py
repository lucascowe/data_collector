import requests

LOCAL = "http://0.0.0.0:5000/track"
NET = "https://track.8500ejefferson.duckdns.org"
SURFACE = "http://192.168.0.137:5000/track"

def main():
    url = LOCAL
    payload = {"test": "payload"}
    response = requests.put(url, json=payload)
    print(f"Response {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
