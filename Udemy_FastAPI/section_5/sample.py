import requests
import json

def main():
    url = 'http://127.0.0.1:8000/'
    data = {
        "x": 2.1,
        "y": 3.3
    }
    res = requests.post(url, data=json.dumps(data))
    print(res.json())
    
if __name__ == '__main__':
    main()
