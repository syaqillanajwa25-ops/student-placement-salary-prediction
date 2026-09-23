import pandas as pd
import numpy as np

def load_data():

    X = pd.read_csv("A.csv")
    y = pd.read_csv("A_targets.csv")

    # gabungkan feature dan target
    df = X.merge(y, on="Student_ID")

    # perbaiki data aneh categorical
    valid = ['Low', 'Medium', 'High']

    df['extracurricular_involvement'] = df[
        'extracurricular_involvement'
    ].apply(lambda x: x if x in valid else np.nan)

    # feature engineering
    df['total_skills'] = (
        df['coding_skill_rating'] +
        df['communication_skill_rating'] +
        df['aptitude_skill_rating']
    )

    # hapus ID
    df.drop(columns=["Student_ID"], inplace=True)

    return df