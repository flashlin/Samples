import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
YOUR_EMAIL = os.getenv("YOUR_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
# https://ironman.atlassian.net/rest/api/2/issue/createmeta/
# https://ironman.atlassian.net/rest/api/2/search?jql=project=B2CS&maxResults=10

# https://ironman.atlassian.net/jira/software/c/projects/B2C/boards/164

# api token from https://id.atlassian.com/manage-profile/security/api-tokens

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
    YOUR_EMAIL, JIRA_API_TOKEN
))

print(f"{resp=}")
print(f"{resp.text=}")