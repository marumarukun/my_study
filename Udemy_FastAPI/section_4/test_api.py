import requests
import json

def main():
    url = 'http://127.0.0.1:8000/item/'
    body = {
            "name": "T-shirt",
            "description": "string",
            "price": 5980,
            "tax": 1.1
            }
    res = requests.post(url, json.dumps(body))
    # 以下でも同じ
    # res = requests.post(url, json=body)
    print(res.json())
    
if __name__ == '__main__':
    main()
