import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

exams = pd.read_csv(os.path.join(DATA_DIR, "university_exams.csv"))

# Weak supervision labels
exams["label"] = (
    (exams["min_percentage"].fillna(0) <= 60) &
    (exams["exam_level"].isin(["UG", "Diploma"]))
).astype(int)

X, y = build_features(exams)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(BASE_DIR, "ml", "exam_ranker.pkl"))
print("✅ ML model trained and saved")
