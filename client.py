import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Student Career Prediction", layout="wide")

st.title("Student Career Prediction System (Decoupled)")
st.caption("Frontend calls FastAPI backend")

API_URL = "http://127.0.0.1:8000/predict"

# =========================
# SIDEBAR (SAMA SEPERTI NO.3)
# =========================
prediction_mode = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Placement Prediction", "Salary Prediction"]
)

st.sidebar.write("Decoupled Architecture (FastAPI + Streamlit)")

# =========================
# INPUT FORM
# =========================
with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
        coding = st.slider("Coding Skill", 1, 10, 7)
        communication = st.slider("Communication Skill", 1, 10, 7)
        aptitude = st.slider("Aptitude Skill", 1, 10, 7)
        internships = st.slider("Internships", 0, 5, 1)
        projects = st.slider("Projects", 0, 10, 2)

    with col2:
        tenth = st.slider("10th Percentage", 0, 100, 75)
        twelfth = st.slider("12th Percentage", 0, 100, 75)
        attendance = st.slider("Attendance", 0, 100, 85)
        study_hours = st.slider("Study Hours", 0, 12, 4)
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        stress = st.slider("Stress Level", 1, 10, 5)

    submit = st.form_submit_button("Predict")

# =========================
# SEND TO FASTAPI
# =========================
if submit:

    payload = {
        "Student_ID": 1001,
        "gender": "Male",
        "branch": "CSE",
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": 0,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,
        "projects_completed": projects,
        "internships_completed": internships,
        "coding_skill_rating": coding,
        "communication_skill_rating": communication,
        "aptitude_skill_rating": aptitude,
        "hackathons_participated": 0,
        "certifications_count": 1,
        "sleep_hours": sleep_hours,
        "stress_level": stress,
        "part_time_job": "No",
        "family_income_level": "Medium",
        "city_tier": 1,
        "internet_access": "Yes",
        "extracurricular_involvement": None
    }

    response = requests.post(API_URL, json=payload)
    result = response.json()

    st.subheader("Prediction Result")

    st.success(f"Placement: {result['placement_prediction']}")
    st.success(f"Salary: {result['salary_prediction']} LPA")