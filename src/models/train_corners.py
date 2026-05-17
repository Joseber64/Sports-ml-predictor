import pandas as pd
import joblib
import os

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "attack_strength",
    "home_form"
]]

y = df["corners_over_9_5"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = CatBoostClassifier(
    verbose=0
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/model_corners.pkl"
)

print("Corners model trained")
