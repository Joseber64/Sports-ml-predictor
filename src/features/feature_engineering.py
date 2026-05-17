import pandas as pd

df = pd.read_csv("data/processed/features.csv")

df["home_form"] = (
    df["home_goals"] * 3
)

df["away_form"] = (
    df["away_goals"] * 3
)

df["attack_strength"] = (
    df["total_goals"] / 2
)

df["defense_strength"] = (
    abs(df["goal_difference"])
)

df.to_csv(
    "data/processed/model_data.csv",
    index=False
)

print(df.head())
print("Feature engineering completed")
