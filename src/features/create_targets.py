import pandas as pd

df = pd.read_csv("data/raw/matches.csv")

# 1X2
df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
df["draw"] = (df["home_goals"] == df["away_goals"]).astype(int)
df["away_win"] = (df["home_goals"] < df["away_goals"]).astype(int)

# Over/Under
df["over_2_5"] = (
    (df["home_goals"] + df["away_goals"]) > 2.5
).astype(int)

# BTTS
df["btts"] = (
    (df["home_goals"] > 0) &
    (df["away_goals"] > 0)
).astype(int)

df.to_csv(
    "data/processed/features.csv",
    index=False
)

print(df.head())
