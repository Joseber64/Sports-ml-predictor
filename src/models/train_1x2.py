import pandas as pd
import joblib
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "home_form",
    "away_form",
    "attack_strength",
    "defense_strength"
]]

y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("1X2 Accuracy:", acc)

joblib.dump(
    model,
    "models/model_1x2.pkl"
)
