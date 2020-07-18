import requests

def main():
    payload = {"test": "payload"}
    response = requests.post("http://0.0.0.0:5000/track", payload)
    print(f"Response {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
