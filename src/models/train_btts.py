import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "goal_difference",
    "total_goals"
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
    "models/btts_model.pkl"
)

print("BTTS model trained")
