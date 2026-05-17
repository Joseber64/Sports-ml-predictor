import pandas as pd
import joblib
import os

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "home_form",
    "away_form",
    "attack_strength"
]]

y = df["btts"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LGBMClassifier()

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/model_btts.pkl"
)

print("BTTS model trained")
