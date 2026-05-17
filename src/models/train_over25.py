import pandas as pd
import joblib
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "home_form",
    "away_form",
    "attack_strength"
]]

y = df["over_2_5"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier()

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/model_over25.pkl"
)

print("Over 2.5 model trained")
