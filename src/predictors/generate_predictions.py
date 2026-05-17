import pandas as pd
import joblib
import os

os.makedirs("data/predictions", exist_ok=True)

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "home_form",
    "away_form",
    "attack_strength",
    "defense_strength"
]]

model_1x2 = joblib.load(
    "models/model_1x2.pkl"
)

predictions = model_1x2.predict(X)

df["prediction_1x2"] = predictions

df.to_csv(
    "data/predictions/predictions.csv",
    index=False
)

print(df.head())
print("Predictions generated")
