import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "age": 45,
    "tenure": 2,
    "monthly_charges": 120,
    "support_calls": 8
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())