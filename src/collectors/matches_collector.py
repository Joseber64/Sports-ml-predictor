import requests
import pandas as pd
import os

API_KEY = os.getenv("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/fixtures"

headers = {
    "x-apisports-key": API_KEY
}

params = {
    "league": 39,
    "season": 2025
}

response = requests.get(url, headers=headers, params=params)

data = response.json()

matches = []

for item in data["response"]:

    fixture = item["fixture"]
    teams = item["teams"]
    goals = item["goals"]

    matches.append({
        "date": fixture["date"],
        "home_team": teams["home"]["name"],
        "away_team": teams["away"]["name"],
        "home_goals": goals["home"],
        "away_goals": goals["away"]
    })

df = pd.DataFrame(matches)

df.to_csv("data/raw/matches.csv", index=False)

print(df.head())
