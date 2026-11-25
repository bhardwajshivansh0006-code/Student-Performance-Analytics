"""
Student Performance Analytics and Prediction System
Run:
    pip install pandas numpy matplotlib scikit-learn
    python student_performance_project.py
The included CSV is synthetic demonstration data, not real student records.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path(__file__).with_name("student_performance_sample.csv")
FEATURES = [
    "Study_Hours_Per_Day",
    "Attendance_Percent",
    "Assignment_Average",
    "Previous_Score",
]
TARGET = "Final_Score"

def load_and_clean_data(path=DATA_FILE):
    data = pd.read_csv(path)
    data = data.drop_duplicates()
    numeric_cols = FEATURES + [TARGET]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())
    data = data.dropna(subset=["Student_ID"])
    return data

def add_performance_category(data):
    data = data.copy()
    data["Performance_Category"] = pd.cut(
        data[TARGET],
        bins=[-0.01, 39.99, 59.99, 74.99, 100],
        labels=["Needs Improvement", "Average", "Good", "Excellent"]
    )
    return data

def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=150, max_depth=8, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }
    return model, X_test, y_test, predictions, metrics

def predict_one(model, study_hours, attendance, assignment, previous_score):
    sample = pd.DataFrame([{
        FEATURES[0]: study_hours,
        FEATURES[1]: attendance,
        FEATURES[2]: assignment,
        FEATURES[3]: previous_score,
    }])
    return float(np.clip(model.predict(sample)[0], 0, 100))

def main():
    data = add_performance_category(load_and_clean_data())
    model, X_test, y_test, predictions, metrics = train_model(data)

    print("\\nSTUDENT PERFORMANCE ANALYTICS AND PREDICTION")
    print("-" * 48)
    print(f"Records analysed: {len(data)}")
    print("\\nPerformance category counts:")
    print(data["Performance_Category"].value_counts().to_string())
    print("\\nModel evaluation (test set):")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print("\\nEnter sample student details to estimate a score.")
    try:
        study = float(input("Study hours per day (0-24): "))
        attendance = float(input("Attendance percentage (0-100): "))
        assignment = float(input("Assignment average (0-100): "))
        previous = float(input("Previous score (0-100): "))
        if not (0 <= study <= 24 and 0 <= attendance <= 100
                and 0 <= assignment <= 100 and 0 <= previous <= 100):
            raise ValueError("Values are outside the accepted range.")
        estimate = predict_one(model, study, attendance, assignment, previous)
        print(f"Estimated final score: {estimate:.1f}/100")
        print("Note: This is an estimate for learning purposes, not an official grade.")
    except ValueError as error:
        print(f"Invalid input: {error}")

    plt.figure(figsize=(7, 4))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.title("Actual vs Predicted Final Scores")
    plt.tight_layout()
    plt.show()

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    importance.plot(kind="barh", title="Feature Importance")
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
