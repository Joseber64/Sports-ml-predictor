import pandas as pd

df = pd.read_csv("data/processed/features.csv")

df["goal_difference"] = (
    df["home_goals"] - df["away_goals"]
)

df["total_goals"] = (
    df["home_goals"] + df["away_goals"]
)

df["home_points"] = df["home_win"] * 3 + df["draw"]
df["away_points"] = df["away_win"] * 3 + df["draw"]

df.to_csv(
    "data/processed/model_data.csv",
    index=False
)

print(df.head())
