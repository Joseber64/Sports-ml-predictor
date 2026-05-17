import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data/processed/model_data.csv")

X = df[[
    "goal_difference",
    "total_goals",
    "home_points",
    "away_points"
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

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

joblib.dump(
    model,
    "models/over25_model.pkl"
)
