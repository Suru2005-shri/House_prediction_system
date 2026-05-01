"""
stacking.py
-----------
Stacked Generalization Ensemble.

Level-0 (base learners) : Linear Regression, Random Forest, XGBoost
Level-1 (meta-learner)  : Ridge Regression

Stacking trains base models on CV folds, feeds out-of-fold predictions
to the meta-learner — capturing the strengths of each base model.
"""

import numpy as np
from sklearn.ensemble         import StackingRegressor, RandomForestRegressor
from sklearn.linear_model     import LinearRegression, Ridge, Lasso
from sklearn.tree             import DecisionTreeRegressor
from sklearn.model_selection  import cross_val_score
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from logger import get_logger

log = get_logger(__name__)


def build_stacking_model(cfg: dict) -> StackingRegressor:
    """
    Construct the stacked ensemble.
    Base estimators use tuned hyperparameters if available; otherwise defaults.
    """
    rs = cfg["models"]["random_state"]

    base_estimators = [
        ("linear", LinearRegression()),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=2, n_jobs=-1, random_state=rs
        )),
        ("xgb", xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=rs, verbosity=0
        )),
    ]

    final = {
        "ridge" : Ridge(alpha=1.0),
        "lasso" : Lasso(alpha=0.1),
        "linear": LinearRegression(),
    }[cfg["stacking"]["final_estimator"]]

    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final,
        cv=cfg["stacking"]["cv_folds"],
        n_jobs=-1,
        passthrough=False,   # set True to also feed raw features to meta-learner
    )
    return stack


def train_stacking_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cfg:     dict,
) -> tuple:
    """
    Fit the stacking ensemble and return (model, y_pred, metrics_dict).
    """
    log.info("Training Stacking Ensemble (LR + RF + XGB → Ridge) …")
    stack = build_stacking_model(cfg)
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    log.info(f"  Stacking — MAE: ${mae:,.0f}  RMSE: ${rmse:,.0f}  R²: {r2:.4f}")

    metrics = {"Model": "Stacking Ensemble", "MAE": mae, "RMSE": rmse, "R2": r2}
    return stack, y_pred, metrics
