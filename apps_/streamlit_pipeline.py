import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "artifacts_"

# Load models
@st.cache_resource
def load_models():
    try:
        with open(MODEL_DIR / "classification_model.pkl", "rb") as f:
            class_model = pickle.load(f)

        with open(MODEL_DIR / "regression_model.pkl", "rb") as f:
            reg_model = pickle.load(f)

        return class_model, reg_model
    
    except FileNotFoundError as e:
        st.error(f"Models are not found: {e}")
        st.stop()

def main():

    st.title("Student Placement Status and Salary Prediction")

    class_model, reg_model = load_models()

    st.subheader("Input Student Data")

    # Features
    branch = st.selectbox("Branch", ["CS", "ECE", "IT", "ME", "CE"])
    cgpa = st.slider("CGPA", 5.0, 10.0, 8.3)
    tenth = st.slider("Tenth Percentage", 50.0, 100.0, 74.7)
    twelfth = st.slider("Twelfth Percentage", 50.0, 100.0, 74.8)
    backlogs = st.number_input("Backlogs", 0.0, 5.0, 0.0)
    study_hours = st.slider("Study Hours/Day", 0.0, 10.0, 4.0)
    attendance = st.slider("Attendance (%)", 44.7, 99.2, 72.5)
    projects = st.number_input("Projects Completed", 0.0, 8.0, 6.0)
    internships = st.number_input("Internships Completed", 0.0, 4.0, 2.0) 
    coding = st.slider("Coding Skill", 1.0, 5.0, 4.0)
    communication = st.slider("Communication Skill", 1.0, 5.0, 3.0)
    aptitude = st.slider("Aptitude Skill", 1.0, 5.0, 4.0)
    hackathons = st.number_input("Hackathons Participated", 0.0, 6.0, 4.0)
    certifications = st.number_input("Certifications Count", 0.0, 9.0, 3.0)
    sleep = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
    stress = st.slider("Stress Level", 1.0, 10.0, 6.0)
    part_time = st.selectbox("Part Time Job", ["Yes", "No"])
    income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet = st.selectbox("Internet Access", ["Yes", "No"])
    extra = st.selectbox("Extracurricular", ["Yes", "No"])

    data = pd.DataFrame([{
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,
        "projects_completed": projects,
        "internships_completed": internships,
        "coding_skill_rating": coding,
        "communication_skill_rating": communication,
        "aptitude_skill_rating": aptitude,
        "hackathons_participated": hackathons,
        "certifications_count": certifications,
        "sleep_hours": sleep,
        "stress_level": stress,
        "part_time_job": part_time,
        "family_income_level": income,
        "city_tier": city,
        "internet_access": internet,
        "extracurricular_involvement": extra
    }])

    st.write("Input Data")
    st.write(data)

    # Prediction
    if st.button("Predict"):

        class_pred = class_model.predict(data)[0]
        reg_pred = reg_model.predict(data)[0]

        st.subheader("Prediction Result")

        if class_pred == 1:
            st.success("Placed")
        else:
            st.error("Not Placed")

        st.info(f"Predicted Salary: {reg_pred:.2f} LPA")

if __name__ == "__main__":
    main()