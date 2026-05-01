"""
data_generator.py
-----------------
Generates a realistic synthetic housing dataset for model training.
Simulates real-world Indian/US housing market patterns.
"""

import numpy as np
import pandas as pd

def generate_housing_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic but realistic housing dataset.

    Features:
        area_sqft      : Total area of the house in square feet
        bedrooms       : Number of bedrooms (1–6)
        bathrooms      : Number of bathrooms (1–4)
        floors         : Number of floors (1–3)
        age_years      : Age of the property in years
        garage         : 1 = has garage, 0 = no garage
        garden         : 1 = has garden, 0 = no garden
        swimming_pool  : 1 = has pool, 0 = no pool
        location_score : Locality rating (1=poor … 10=prime)
        furnishing     : 0=unfurnished, 1=semi, 2=fully furnished
        distance_city  : Distance from city centre in km
        school_nearby  : 1 = good school nearby, 0 = no
        price_usd      : Target variable – house price in USD

    Returns:
        pd.DataFrame with n_samples rows
    """
    rng = np.random.RandomState(random_state)

    # ── Core structural features ──────────────────────────────────────────
    area_sqft    = rng.randint(600, 5500, n_samples).astype(float)
    bedrooms     = rng.choice([1, 2, 3, 4, 5, 6], n_samples,
                              p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05])
    bathrooms    = np.clip(bedrooms - rng.choice([0, 1], n_samples, p=[0.6, 0.4]), 1, 4)
    floors       = rng.choice([1, 2, 3], n_samples, p=[0.50, 0.35, 0.15])
    age_years    = rng.randint(0, 51, n_samples).astype(float)

    # ── Amenity flags ─────────────────────────────────────────────────────
    garage       = rng.binomial(1, 0.55, n_samples)
    garden       = rng.binomial(1, 0.40, n_samples)
    swimming_pool= rng.binomial(1, 0.15, n_samples)

    # ── Location features ─────────────────────────────────────────────────
    location_score = np.clip(rng.normal(5.5, 2.0, n_samples), 1, 10).round(1)
    distance_city  = np.clip(rng.exponential(10, n_samples), 0.5, 50).round(1)
    school_nearby  = (location_score >= 6).astype(int)

    # ── Furnishing ────────────────────────────────────────────────────────
    furnishing   = rng.choice([0, 1, 2], n_samples, p=[0.30, 0.45, 0.25])

    # ── Price formula (reflects real market logic) ─────────────────────
    base_price = (
          80  * area_sqft
        + 8_000  * bedrooms
        + 12_000 * bathrooms
        + 10_000 * floors
        - 800    * age_years
        + 15_000 * garage
        + 10_000 * garden
        + 40_000 * swimming_pool
        + 18_000 * location_score
        - 2_500  * distance_city
        + 8_000  * school_nearby
        + 6_000  * furnishing
    )

    # Add realistic noise (±12 %)
    noise       = rng.normal(0, 0.12 * base_price, n_samples)
    price_usd   = np.clip(base_price + noise, 50_000, 2_500_000).round(-2)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "area_sqft"     : area_sqft,
        "bedrooms"      : bedrooms,
        "bathrooms"     : bathrooms,
        "floors"        : floors,
        "age_years"     : age_years,
        "garage"        : garage,
        "garden"        : garden,
        "swimming_pool" : swimming_pool,
        "location_score": location_score,
        "distance_city" : distance_city,
        "school_nearby" : school_nearby,
        "furnishing"    : furnishing,
        "price_usd"     : price_usd,
    })

    return df


if __name__ == "__main__":
    df = generate_housing_data()
    print(f"Dataset shape : {df.shape}")
    print(df.head())
    print("\nBasic stats:")
    print(df.describe().round(2))
