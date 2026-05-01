"""
models.py
---------
Train, evaluate, and compare multiple regression models.
Models: Linear Regression | Decision Tree | Random Forest | XGBoost
"""

import numpy as np
from sklearn.linear_model  import LinearRegression
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
import xgboost             as xgb
import joblib
import os


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : ${mae:>12,.0f}")
    print(f"    RMSE : ${rmse:>12,.0f}")
    print(f"    R²   : {r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ── Individual trainers ───────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train, X_test, y_test) -> tuple:
    """Baseline: Ordinary Least Squares Linear Regression."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model("Linear Regression", y_test, y_pred)
    return model, y_pred, metrics


def train_decision_tree(X_train, y_train, X_test, y_test) -> tuple:
    """Decision Tree Regressor — interpretable, but prone to overfit."""
    model = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model("Decision Tree", y_test, y_pred)
    return model, y_pred, metrics


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    """Random Forest — ensemble of trees, robust and accurate."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model("Random Forest", y_test, y_pred)
    return model, y_pred, metrics


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    """XGBoost — gradient boosted trees, often top performer."""
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model("XGBoost", y_test, y_pred)
    return model, y_pred, metrics


# ── Master trainer ────────────────────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Train all four regression models and return results dict.

    Returns:
        {
          "Linear Regression" : (model, y_pred, metrics),
          "Decision Tree"     : (model, y_pred, metrics),
          "Random Forest"     : (model, y_pred, metrics),
          "XGBoost"           : (model, y_pred, metrics),
        }
    """
    print("\n" + "="*55)
    print("          MODEL TRAINING & EVALUATION")
    print("="*55)

    results = {}
    results["Linear Regression"] = train_linear_regression(X_train, y_train, X_test, y_test)
    results["Decision Tree"]     = train_decision_tree    (X_train, y_train, X_test, y_test)
    results["Random Forest"]     = train_random_forest    (X_train, y_train, X_test, y_test)
    results["XGBoost"]           = train_xgboost          (X_train, y_train, X_test, y_test)

    return results


# ── Save best model ───────────────────────────────────────────────────────────

def save_best_model(results: dict, save_dir: str = "models") -> str:
    """Pick the model with the highest R² and save it with joblib."""
    os.makedirs(save_dir, exist_ok=True)

    best_name = max(results, key=lambda k: results[k][2]["R2"])
    best_model = results[best_name][0]

    path = os.path.join(save_dir, "best_model.pkl")
    joblib.dump(best_model, path)
    print(f"\n✅ Best model : {best_name}  (saved → {path})")
    return best_name
