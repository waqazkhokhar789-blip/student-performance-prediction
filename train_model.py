from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_model.pkl"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_completed",
    "sleep_hours",
]

TARGET = "exam_score"


def main():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_PATH
    )

    print("Model training successful!")
    print("Total student records:", len(df))
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared score: {r2:.2f}")
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
