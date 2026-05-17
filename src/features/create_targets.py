import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/matches.csv")

# 1X2
df["result"] = df.apply(
    lambda row: 1 if row["home_goals"] > row["away_goals"]
    else 2 if row["home_goals"] < row["away_goals"]
    else 0,
    axis=1
)

# Over 2.5
df["over_2_5"] = (
    df["total_goals"] > 2.5
).astype(int)

# BTTS
df["btts"] = (
    (df["home_goals"] > 0) &
    (df["away_goals"] > 0)
).astype(int)

# Corners
df["corners_over_9_5"] = (
    df["corners"] > 9.5
).astype(int)

# Cards
df["cards_over_3_5"] = (
    df["cards"] > 3.5
).astype(int)

df.to_csv(
    "data/processed/features.csv",
    index=False
)

print(df.head())
print("Targets created")
