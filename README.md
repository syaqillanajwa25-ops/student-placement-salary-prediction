# Student Placement & Salary Prediction

An end-to-end machine learning project for predicting student placement status and estimating expected salary based on academic performance, technical skills, and other student-related attributes.

The project includes data preprocessing, classification and regression modeling, hyperparameter optimization, experiment tracking, and model serving using Streamlit and FastAPI.

## Project Overview

This project addresses two machine learning tasks:

1. **Placement Prediction**  
   Predicts whether a student is likely to be **Placed** or **Not Placed**.

2. **Salary Prediction**  
   Estimates the expected salary of a student using a regression model.

The models use student information such as CGPA, academic performance, coding and communication skills, internships, projects, attendance, and other related features.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Optuna
- MLflow
- FastAPI
- Streamlit
- Uvicorn

## Machine Learning Pipeline

The project includes:

- Data ingestion and preprocessing
- Numerical feature imputation and scaling
- Categorical feature imputation and one-hot encoding
- Train-test splitting
- Random Forest classification for placement prediction
- Random Forest regression for salary prediction
- Hyperparameter tuning using Optuna
- Experiment tracking using MLflow
- Model serialization using Pickle

## Models

### Placement Prediction

A **Random Forest Classifier** is used to predict whether a student will be placed.

Optuna is used to optimize model hyperparameters, including:

- Number of estimators
- Maximum tree depth

The trained placement model is stored as:

```text
best_model.pkl
