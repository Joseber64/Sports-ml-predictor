import pandas as pd
import joblib
import os

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "defense_strength",
    "attack_strength"
]]

y = df["cards_over_3_5"]

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
    "models/model_cards.pkl"
)

print("Cards model trained")
