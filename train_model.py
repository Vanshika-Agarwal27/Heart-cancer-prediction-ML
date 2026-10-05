import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "heart_disease.csv")
MODEL_PATH = os.path.join(BASE, "models", "heart_disease_model.pkl")

df = pd.read_csv(DATA_PATH)
features = list(df.columns[:-1])
X = df[features]
y = df["heart_disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250, max_depth=10, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Model saved successfully.")
