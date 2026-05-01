"""
predictor.py
------------
Predict house price for a single new property.
Applies the same feature engineering done during training.
"""

import numpy as np


FURNISHING_MAP = {"unfurnished": 0, "semi-furnished": 1, "fully-furnished": 2}


def build_input_vector(user_input: dict, scaler, feature_names: list) -> np.ndarray:
    """
    Convert raw user-provided property details into a scaled feature vector.

    user_input keys (all required):
        area_sqft, bedrooms, bathrooms, floors, age_years,
        garage, garden, swimming_pool, location_score,
        distance_city, school_nearby, furnishing (int 0/1/2)
    """
    ui = user_input

    # ── Replicate feature engineering (must match preprocessor.py) ─────────
    total_rooms   = ui["bedrooms"] + ui["bathrooms"]
    amenity_score = ui["garage"] + ui["garden"] + ui["swimming_pool"] * 2

    age = ui["age_years"]
    if age <= 5:
        age_category = 3
    elif age <= 15:
        age_category = 2
    elif age <= 30:
        age_category = 1
    else:
        age_category = 0

    ls = ui["location_score"]
    if ls <= 3.5:
        location_tier = 0
    elif ls <= 6.5:
        location_tier = 1
    else:
        location_tier = 2

    # ── Build raw feature vector in same column order as training ──────────
    feature_dict = {
        "area_sqft"     : ui["area_sqft"],
        "bedrooms"      : ui["bedrooms"],
        "bathrooms"     : ui["bathrooms"],
        "floors"        : ui["floors"],
        "age_years"     : ui["age_years"],
        "garage"        : ui["garage"],
        "garden"        : ui["garden"],
        "swimming_pool" : ui["swimming_pool"],
        "location_score": ui["location_score"],
        "distance_city" : ui["distance_city"],
        "school_nearby" : ui["school_nearby"],
        "furnishing"    : ui["furnishing"],
        "total_rooms"   : total_rooms,
        "amenity_score" : amenity_score,
        "age_category"  : age_category,
        "location_tier" : location_tier,
    }

    # Align to training column order
    vector = np.array([feature_dict[f] for f in feature_names], dtype=float).reshape(1, -1)
    return scaler.transform(vector)


def predict_price(model, vector: np.ndarray) -> float:
    """Return predicted price (USD) from the trained model."""
    return float(model.predict(vector)[0])


def format_price(price: float) -> str:
    """Format price as human-readable USD string."""
    if price >= 1_000_000:
        return f"${price/1_000_000:.2f}M  (${price:,.0f})"
    return f"${price/1_000:.1f}K  (${price:,.0f})"


def print_prediction_report(user_input: dict, predicted_price: float, model_name: str):
    """Pretty-print a prediction summary to the terminal."""
    furnishing_labels = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Fully Furnished"}
    print("\n" + "="*55)
    print("        🏠  HOUSE PRICE PREDICTION REPORT")
    print("="*55)
    print(f"  Area           : {user_input['area_sqft']:,} sq ft")
    print(f"  Bedrooms       : {user_input['bedrooms']}")
    print(f"  Bathrooms      : {user_input['bathrooms']}")
    print(f"  Floors         : {user_input['floors']}")
    print(f"  Age            : {user_input['age_years']} years")
    print(f"  Garage         : {'Yes' if user_input['garage'] else 'No'}")
    print(f"  Garden         : {'Yes' if user_input['garden'] else 'No'}")
    print(f"  Swimming Pool  : {'Yes' if user_input['swimming_pool'] else 'No'}")
    print(f"  Location Score : {user_input['location_score']}/10")
    print(f"  Distance City  : {user_input['distance_city']} km")
    print(f"  School Nearby  : {'Yes' if user_input['school_nearby'] else 'No'}")
    print(f"  Furnishing     : {furnishing_labels[user_input['furnishing']]}")
    print("-"*55)
    print(f"  🤖 Model Used  : {model_name}")
    print(f"  💰 Predicted   : {format_price(predicted_price)}")
    print("="*55)
