import requests

try:
    response = requests.post('http://127.0.0.1:5000/', data={'url': 'http://google.com'})
    print(f"Status Code: {response.status_code}")
    if response.status_code == 405:
        print("Error: Method Not Allowed (405)")
    elif response.status_code == 200:
        print("Success: POST request accepted")
    else:
        print(f"Unexpected status: {response.status_code}")
except Exception as e:
    print(f"Connection failed: {e}")
