import requests
import json

# https://ironman.atlassian.net/rest/api/2/issue/createmeta/
# https://ironman.atlassian.net/rest/api/2/search?jql=project=B2CS&maxResults=10

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
        "summary": "",
        "description": "",
        "issuetype": {
            "name": "Task"
        }
    }
})

resp = requests.get(url, headers=headers, data=payload, auth=(
    "your_email", "api token"
))

print(f"{resp=}")
print(f"{resp.text=}")