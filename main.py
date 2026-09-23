import pandas as pd
import numpy as np
import pickle
import optuna
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# load data
df = pd.read_csv("A.csv")
target = pd.read_csv("A_targets.csv")

# gabungkan target
df["Placement_Status"] = target.iloc[:, 0]
df["Salary"] = target.iloc[:, 1]

# pisahkan feature dan target
X = df.drop(columns=["Placement_Status", "Salary"])
y_class = df["Placement_Status"]
y_reg = df["Salary"]

# split data
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# kolom numerik dan kategori
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# preprocessing
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# optuna tuning
def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 3, 15)

    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train_class)
    pred = model.predict(X_test)

    return accuracy_score(y_test_class, pred)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best = study.best_params

# model klasifikasi
placement_model = Pipeline([
    ("prep", prep),
    ("model", RandomForestClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        random_state=42
    ))
])

placement_model.fit(X_train, y_train_class)
pred_class = placement_model.predict(X_test)
acc = accuracy_score(y_test_class, pred_class)

# model regresi
salary_model = Pipeline([
    ("prep", prep),
    ("model", RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    ))
])

salary_model.fit(X_train, y_train_reg)
pred_reg = salary_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))

# simpan model
with open("best_model.pkl", "wb") as f:
    pickle.dump(placement_model, f)

with open("salary_model.pkl", "wb") as f:
    pickle.dump(salary_model, f)

# mlflow
with mlflow.start_run():

    mlflow.log_param("classifier", "RandomForest")
    mlflow.log_param("regressor", "RandomForest")
    mlflow.log_param("n_estimators", best["n_estimators"])
    mlflow.log_param("max_depth", best["max_depth"])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(placement_model, "placement_model")
    mlflow.sklearn.log_model(salary_model, "salary_model")

print("Classification Accuracy:", round(acc, 3))
print("Regression RMSE:", round(rmse, 2))
print("Models saved successfully.")
print("MLflow logged successfully.")
