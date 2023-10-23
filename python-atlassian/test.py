import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://<your-domain.atlassian.net>/wiki/rest/api/space"
url = "https://ironman.atlassian.net/wiki/rest/api/spaces/B2C/overview"

headers = {
    "Authorization": f"Basic {os.getenv('ATLASSIAN_ACCESS_TOKEN')}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Request failed with status code: {response.status_code}")
