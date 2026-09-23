from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

with open("best_model.pkl", "rb") as f:
    placement_model = pickle.load(f)

with open("salary_model.pkl", "rb") as f:
    salary_model = pickle.load(f)


class StudentInput(BaseModel):
    cgpa: float
    coding_skill_rating: int
    communication_skill_rating: int


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: StudentInput):

    df = pd.DataFrame([{
        "Student_ID": 1001,
        "gender": "Male",
        "branch": "CSE",
        "cgpa": float(data.cgpa),
        "tenth_percentage": 80.0,
        "twelfth_percentage": 80.0,
        "backlogs": 0,
        "study_hours_per_day": 4.0,
        "attendance_percentage": 90.0,
        "projects_completed": 2,
        "internships_completed": 1,
        "coding_skill_rating": int(data.coding_skill_rating),
        "communication_skill_rating": int(data.communication_skill_rating),
        "aptitude_skill_rating": 7,
        "hackathons_participated": 0,
        "certifications_count": 1,
        "sleep_hours": 7.0,
        "stress_level": 5,
        "part_time_job": "No",
        "family_income_level": "Medium",
        "city_tier": 1,
        "internet_access": "Yes",
        "extracurricular_involvement": "Medium"
    }])

    placement = placement_model.predict(df)[0]

    if str(placement).lower() == "placed":
        salary = round(float(salary_model.predict(df)[0]), 2)
    else:
        salary = 0

    return {
        "placement_prediction": str(placement),
        "salary_prediction": salary
    }
