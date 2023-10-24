import requests
import json

url = "https://ironman.atlassian.net/rest/api/3/project"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}
payload = json.dumps({
    "fields": {
        "project": {
            "key": "B2CS"
        },
    }
})

resp = requests.get(url, headers=headers, data=payload, auth=(
    "your_email", "api token"
))

print(f"{resp=}")
print(f"{resp.text=}")