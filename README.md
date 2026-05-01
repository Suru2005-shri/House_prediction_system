#  House Price Prediction using Regression Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An end-to-end machine learning project that predicts house prices using regression models — built as a student portfolio project demonstrating real-world Data Science skills.

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Industry Relevance](#-industry-relevance)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#️-project-architecture)
- [Dataset](#-dataset)
- [Models Used](#-models-used)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [How to Run](#-how-to-run)
- [Screenshots & Outputs](#-screenshots--outputs)
- [Key Learnings](#-key-learnings)
- [Interview Prep](#-interview-prep)

---

##  Project Overview

This project builds a complete **House Price Prediction system** using multiple regression algorithms. Given features of a property (area, bedrooms, location, age, amenities), the model predicts the market price in USD.

**Simple Explanation:**  
Just like how a doctor estimates your health based on symptoms, this project estimates a house's value based on its features — area, number of rooms, location quality, age, and amenities.

**Technical Explanation:**  
We apply supervised regression machine learning. The model learns the mathematical relationship between 16 input features and the target variable (price_usd) by minimizing prediction error across 1,598 training samples.

**Workflow:**
```
Raw Housing Data → Data Cleaning → Feature Engineering → Model Training → Price Prediction → Insights
```

---

##  Problem Statement

Real estate markets involve millions of transactions. Accurately pricing a property is critical for:
- **Buyers** — avoid overpaying
- **Sellers** — avoid underpricing
- **Banks/Lenders** — assess loan risk (LTV ratio)
- **Investors** — identify undervalued properties
- **Property Portals** — auto-generate price estimates

Manual appraisal is slow, expensive, and inconsistent. A trained ML model can predict prices instantly and at scale.

---

##  Industry Relevance

| Sector | How They Use It |
|--------|----------------|
|  **Banks** | Determine property value for home loan approvals |
|  **Real Estate Portals** | Show estimated prices (like Zillow Zestimate) |
|  **Investment Firms** | Find undervalued properties for ROI |
|  **Builders/Developers** | Price new developments competitively |
|  **Government** | Property tax assessment and urban planning |

---

##  Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.8+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Boosting | XGBoost |
| Model Persistence | Joblib |
| Notebook | Jupyter |

---

##  Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  area_sqft | bedrooms | bathrooms | floors | age_years  │
│  garage | garden | pool | location_score | distance     │
│  school_nearby | furnishing                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 PREPROCESSING                           │
│  • Drop duplicates & outliers (IQR method)              │
│  • Fill missing values with median                      │
│  • Feature Engineering (4 new features)                 │
│  • StandardScaler normalization                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              REGRESSION MODELS                          │
│  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │ Linear Regression│  │    Decision Tree          │   │
│  └──────────────────┘  └───────────────────────────┘   │
│  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │  Random Forest   │  │        XGBoost            │   │
│  └──────────────────┘  └───────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                OUTPUT LAYER                             │
│             Predicted House Price (USD)               │
└─────────────────────────────────────────────────────────┘
```

---

##  Dataset

| Property | Value |
|----------|-------|
| Type | Synthetic (programmatically generated) |
| Samples | 2,000 rows |
| Features | 12 raw + 4 engineered = 16 total |
| Target | `price_usd` |
| Price Range | ~$50,000 – $2,500,000 |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `area_sqft` | Numeric | Property area in square feet |
| `bedrooms` | Ordinal | Number of bedrooms (1–6) |
| `bathrooms` | Ordinal | Number of bathrooms (1–4) |
| `floors` | Ordinal | Number of floors (1–3) |
| `age_years` | Numeric | Property age in years |
| `garage` | Binary | 1 = has garage |
| `garden` | Binary | 1 = has garden |
| `swimming_pool` | Binary | 1 = has pool |
| `location_score` | Numeric | Locality rating (1–10) |
| `distance_city` | Numeric | Distance from city centre (km) |
| `school_nearby` | Binary | 1 = good school nearby |
| `furnishing` | Ordinal | 0=unfurnished, 1=semi, 2=fully |
| `total_rooms`  | Engineered | bedrooms + bathrooms |
| `amenity_score`  | Engineered | Composite luxury score |
| `age_category`  | Engineered | New(3)/Recent(2)/Old(1)/Very Old(0) |
| `location_tier`  | Engineered | Budget(0)/Mid(1)/Premium(2) |

---

##  Models Used

### 1. Linear Regression (Baseline)
- Fits a straight-line relationship between features and price
- Fast, interpretable, good baseline
- **Best for:** Understanding which features drive price

### 2. Decision Tree Regressor
- Splits data based on feature thresholds
- Highly interpretable
- **Best for:** Understanding decision logic

### 3. Random Forest Regressor
- Ensemble of 200 decision trees
- Robust to noise and overfitting
- **Best for:** Accurate general predictions

### 4. XGBoost Regressor
- Gradient boosted trees
- Industry standard for tabular data
- **Best for:** Maximum accuracy

---

## 📈 Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** ⭐ | $36,916 | $48,551 | **0.8694** |
| XGBoost | $39,833 | $51,818 | 0.8513 |
| Random Forest | $41,387 | $53,819 | 0.8396 |
| Decision Tree | $51,912 | $66,444 | 0.7555 |

>  **Best model: Linear Regression** with R² = 0.8694  
> Meaning: the model explains **86.94%** of the variance in house prices.

### Sample Prediction
```
Property: 2200 sq ft | 4 BR | 3 BA | 2 Floors | 8yr | Garage ✓ | Garden ✓
Location: Score 7.5/10 | 6km from city | School nearby

 Predicted Price: $421,442
```

---

##  Project Structure

```
House-Price-Prediction/
│
├── data/
│   ├── housing_data.csv           # Raw synthetic dataset
│   └── housing_engineered.csv    # After feature engineering
│
├── notebooks/
│   └── house_price_prediction.ipynb  # Interactive Jupyter notebook
│
├── src/
│   ├── data_generator.py         # Generates synthetic housing data
│   ├── preprocessor.py           # Cleaning, engineering, scaling
│   ├── models.py                 # All 4 regression models
│   ├── visualizer.py             # All charts & plots
│   └── predictor.py              # Price prediction for new input
│
├── models/
│   └── best_model.pkl            # Saved best trained model
│
├── outputs/
│   ├── 01_data_overview.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_distribution.png
│   ├── 04_feature_vs_price.png
│   ├── 05_categorical_analysis.png
│   ├── 06_model_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_residual_analysis.png
│   └── 10_price_trend_by_area.png
│
├── main.py                       # ← Run this to execute everything
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup (Windows / Mac / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  How to Run

### Option A — Run Full Pipeline (Recommended)
```bash
python main.py
```

**Expected Terminal Output:**
```
🏠  HOUSE PRICE PREDICTION  |  ML Pipeline

PHASE 1 │ Generating Dataset         
PHASE 2 │ Data Cleaning              
PHASE 3 │ Feature Engineering        
PHASE 4 │ EDA Visualizations         
PHASE 5 │ Train-Test Split           
PHASE 6 │ Model Training             
PHASE 7 │ Save Best Model            
PHASE 8 │ Evaluation Plots           
PHASE 9 │ Price Prediction           

 Predicted Price: $421,442
```

### Option B — Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

### Customise the Prediction
Edit `SAMPLE_HOUSE` in `main.py`:
```python
SAMPLE_HOUSE = {
    "area_sqft"     : 1500,
    "bedrooms"      : 3,
    "bathrooms"     : 2,
    "floors"        : 1,
    "age_years"     : 10,
    "garage"        : 1,
    "garden"        : 0,
    "swimming_pool" : 0,
    "location_score": 6.5,
    "distance_city" : 8.0,
    "school_nearby" : 1,
    "furnishing"    : 0,
}
```

---

## Screenshots & Outputs

All charts are automatically saved to the `outputs/` folder.

| File | Description |
|------|-------------|
| `01_data_overview.png` | Histogram of all features |
| `02_correlation_heatmap.png` | Correlation matrix between all features |
| `03_price_distribution.png` | House price distribution (raw + log scale) |
| `04_feature_vs_price.png` | Scatter plots: each feature vs price |
| `05_categorical_analysis.png` | Box plots: price by category |
| `06_model_comparison.png` | MAE, RMSE, R² comparison bar charts |
| `07_actual_vs_predicted.png` | Actual vs predicted scatter for all models |
| `08_residual_analysis.png` | Residual distribution for best model |
| `10_price_trend_by_area.png` | Price vs area split by bedrooms |

---

##  Key Learnings

1. **Data Generation** — Simulating realistic datasets when real data is unavailable
2. **EDA** — Finding patterns, correlations, and outliers before modelling
3. **Feature Engineering** — Creating new features that boost model performance
4. **Regression Models** — Understanding Linear Regression, Decision Trees, Random Forest, XGBoost
5. **Model Evaluation** — Using MAE, RMSE, R² to measure and compare models
6. **ML Pipeline** — Building modular, reusable, production-style code
7. **GitHub Portfolio** — Structuring a project for professional presentation

---

##  Interview Prep

**Q: Why did you choose regression over classification?**  
A: House price is a continuous variable (not a category), so regression is the correct problem type.

**Q: What does R² = 0.87 mean?**  
A: The model explains 87% of the variance in house prices. The remaining 13% is due to factors not captured in our data.

**Q: Why does Linear Regression outperform XGBoost here?**  
A: The synthetic data was generated using a linear formula, so linear models fit it perfectly. Real-world data is more complex, where XGBoost typically wins.

**Q: What is feature engineering?**  
A: Creating new meaningful features from existing ones. E.g., `total_rooms = bedrooms + bathrooms` gives the model a more useful signal.

**Q: What is StandardScaler and why is it needed?**  
A: It normalizes features to mean=0, std=1. Without it, features with large values (area_sqft in thousands) would dominate features with small values (garage = 0 or 1).

**Q: What are the business applications of this model?**  
A: Banks use it for loan appraisals, portals like Zillow/MagicBricks use it for auto-valuations, investors use it to find underpriced properties.

---

##  License

MIT License — free to use for educational and portfolio purposes.

---

## 🤝 Connect

**Built by:**Shruti Srivastava
- GitHub:(https://github.com/Suru2005-shri)
- LinkedIn: www.linkedin.com/in/shruti-srivastava-36b26232a
<p align="center">
  <img src="https://github.com/Suru2005-shri/House_prediction_system/blob/6da89f24bc19e52b3635659dbda596569a55f8e6/outputs/00_pro_dashboard.png" />
  
</p>


*If this project helped you, please ⭐ star the repo!*
