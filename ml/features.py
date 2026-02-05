from sklearn.preprocessing import LabelEncoder
import pandas as pd

def build_features(exam_df):
    df = exam_df.copy()

    df["min_percentage"] = df["min_percentage"].fillna(0)
    df["application_fee"] = df["application_fee"].fillna(0)

    cat_cols = ["stream", "exam_level", "state"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    features = [
        "min_percentage",
        "application_fee",
        "stream",
        "exam_level",
        "state"
    ]

    return df[features], df["label"]
