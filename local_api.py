import os
import requests

# Ensure localhost calls don't use a proxy
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

BASE_URL = "http://127.0.0.1:8000"

# Create a session that ignores environment proxy settings
session = requests.Session()
session.trust_env = False

# Test GET request
r = session.get(BASE_URL + "/", timeout=5)
print("GET / ->", r.status_code)
if r.status_code == 200:
    print("Response:", r.json())
else:
    print("GET Error Text:", r.text)

# Data for POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Test POST request
r = session.post(BASE_URL + "/predict", json=data, timeout=5)
print("\nPOST /predict ->", r.status_code)
if r.status_code == 200:
    print("Result:", r.json())
else:
    print("POST Error Text:", r.text)
