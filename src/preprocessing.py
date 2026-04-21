from pathlib import Path
import pandas as pd
import pickle

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ingested_" / "data.csv"

def preprocess_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_csv(INPUT_FILE)

    # Encode target variable (categorical)
    df["placement_status"] = df["placement_status"].map({
        "Placed": 1,
        "Not Placed": 0
    })

    # Feature selection
    feature_columns = [
        "branch",
        "part_time_job",
        "family_income_level",
        "city_tier",
        "internet_access",
        "extracurricular_involvement",
        "cgpa",
        "tenth_percentage",
        "twelfth_percentage",
        "backlogs",
        "study_hours_per_day",
        "attendance_percentage",
        "projects_completed",
        "internships_completed",
        "coding_skill_rating",
        "communication_skill_rating",
        "aptitude_skill_rating",
        "hackathons_participated",
        "certifications_count",
        "sleep_hours",
        "stress_level"
    ]

    X = df[feature_columns].copy()
    y_class = df["placement_status"]
    y_reg = df["salary_lpa"]

    print("Preprocessing completed.")

    return X, y_class, y_reg

if __name__ == "__main__":
    preprocess_data()