import requests
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

API_KEY = os.getenv("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/fixtures"

headers = {
    "x-apisports-key": API_KEY
}

params = {
    "league": 39,
    "season": 2025
}

response = requests.get(
    url,
    headers=headers,
    params=params
)

data = response.json()

matches = []

for item in data["response"]:

    fixture = item["fixture"]
    teams = item["teams"]
    goals = item["goals"]

    home_goals = goals["home"]
    away_goals = goals["away"]

    if home_goals is None:
        continue

    matches.append({
        "date": fixture["date"],
        "home_team": teams["home"]["name"],
        "away_team": teams["away"]["name"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "total_goals": home_goals + away_goals,
        "goal_difference": home_goals - away_goals,
        "corners": 10,
        "cards": 4
    })

df = pd.DataFrame(matches)

df.to_csv(
    "data/raw/matches.csv",
    index=False
)

print(df.head())
print("Matches collected successfully")
