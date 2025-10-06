
import requests

url = "http://localhost:8000/doc_generator/api/get-water-quality-data/?lat=-45.8833&lon=168.3667"

try:
    response = requests.get(url)
    response.raise_for_status()
    print(response.text)
except requests.exceptions.RequestException as e:
    print(f"Failed to fetch data: {e}")
