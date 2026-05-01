"""
preprocessor.py
---------------
Data cleaning, feature engineering, and scaling pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 – Clean raw housing data.
    - Drop duplicates
    - Handle missing values
    - Remove price outliers using IQR method
    """
    print("\n[Preprocessing] Starting data cleaning...")
    original_shape = df.shape

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill numeric nulls with median (safe for skewed data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Remove price outliers (keep within 1.5 * IQR)
    Q1 = df["price_usd"].quantile(0.25)
    Q3 = df["price_usd"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["price_usd"] >= Q1 - 1.5 * IQR) &
            (df["price_usd"] <= Q3 + 1.5 * IQR)]

    print(f"  Rows before cleaning : {original_shape[0]}")
    print(f"  Rows after  cleaning : {df.shape[0]}")
    print(f"  Columns              : {df.shape[1]}")
    return df.reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 – Create new features from existing ones.
    These derived features often improve model accuracy.
    """
    print("\n[Preprocessing] Engineering new features...")

    # Price per sqft (a standard real-estate metric)
    df["price_per_sqft"] = (df["price_usd"] / df["area_sqft"]).round(2)

    # Total rooms = bedrooms + bathrooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    # Amenity score (composite luxury index)
    df["amenity_score"] = (df["garage"] + df["garden"] + df["swimming_pool"] * 2)

    # Property age category
    df["age_category"] = pd.cut(
        df["age_years"],
        bins=[-1, 5, 15, 30, 50],
        labels=[3, 2, 1, 0]          # newer = higher score
    ).astype(int)

    # Location tier (based on location_score)
    df["location_tier"] = pd.cut(
        df["location_score"],
        bins=[0, 3.5, 6.5, 10],
        labels=[0, 1, 2]             # 0=budget, 1=mid, 2=premium
    ).astype(int)

    print(f"  Features added: price_per_sqft, total_rooms, amenity_score, "
          f"age_category, location_tier")
    return df


def prepare_features(df: pd.DataFrame, target: str = "price_usd"):
    """
    Step 3 – Split features / target and apply StandardScaler.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    print("\n[Preprocessing] Preparing features for modelling...")

    # Drop derived price column (leakage risk) and target
    drop_cols = [target, "price_per_sqft"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target].values

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing  samples : {X_test.shape[0]}")
    print(f"  Feature count    : {X_train.shape[1]}")
    print(f"  Features used    : {feature_cols}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols
