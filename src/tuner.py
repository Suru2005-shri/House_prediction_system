"""
tuner.py
--------
Hyperparameter tuning with RandomizedSearchCV + k-Fold cross-validation.
Produces tuned Random Forest and XGBoost models with CV score reports.
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection  import RandomizedSearchCV, KFold, cross_validate
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import make_scorer, mean_absolute_error, r2_score
import xgboost as xgb

from logger import get_logger

log = get_logger(__name__)


# ── Scoring dict used by cross_validate ───────────────────────────────────────
SCORING = {
    "r2"  : make_scorer(r2_score),
    "mae" : make_scorer(mean_absolute_error, greater_is_better=False),
}


def _report_cv(name: str, cv_results: dict) -> dict:
    """Pretty-print cross-validation results and return summary dict."""
    r2_scores  =  cv_results["test_r2"]
    mae_scores = -cv_results["test_mae"]        # negate (stored as negative)
    log.info(f"  CV Results — {name}")
    log.info(f"    R²  : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}  "
             f"[{r2_scores.min():.4f} – {r2_scores.max():.4f}]")
    log.info(f"    MAE : ${mae_scores.mean():,.0f} ± ${mae_scores.std():,.0f}")
    return {
        "model"   : name,
        "r2_mean" : r2_scores.mean(),
        "r2_std"  : r2_scores.std(),
        "r2_scores": r2_scores.tolist(),
        "mae_mean": mae_scores.mean(),
        "mae_std" : mae_scores.std(),
    }


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> tuple:
    """
    RandomizedSearchCV on Random Forest.
    Returns (best_estimator, best_params, cv_summary_dict).
    """
    log.info("Tuning Random Forest with RandomizedSearchCV …")
    t0 = time.time()

    param_dist = {
        "n_estimators"    : cfg["tuning"]["random_forest_grid"]["n_estimators"],
        "max_depth"       : cfg["tuning"]["random_forest_grid"]["max_depth"],
        "min_samples_split": cfg["tuning"]["random_forest_grid"]["min_samples_split"],
        "min_samples_leaf" : cfg["tuning"]["random_forest_grid"]["min_samples_leaf"],
        "max_features"    : cfg["tuning"]["random_forest_grid"]["max_features"],
    }

    base = RandomForestRegressor(random_state=cfg["models"]["random_state"], n_jobs=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=cfg["tuning"]["n_iter"],
        cv=cfg["tuning"]["cv_folds"],
        scoring=cfg["tuning"]["scoring"],
        n_jobs=cfg["tuning"]["n_jobs"],
        random_state=cfg["models"]["random_state"],
        verbose=0,
    )
    search.fit(X_train, y_train)

    log.info(f"  Best params : {search.best_params_}")
    log.info(f"  Best CV R²  : {search.best_score_:.4f}  ({time.time()-t0:.1f}s)")

    # Full cross-validation report on the best estimator
    kf = KFold(n_splits=cfg["tuning"]["cv_folds"], shuffle=True,
               random_state=cfg["models"]["random_state"])
    cv_res = cross_validate(search.best_estimator_, X_train, y_train,
                            cv=kf, scoring=SCORING, return_train_score=False)
    summary = _report_cv("Tuned Random Forest", cv_res)
    return search.best_estimator_, search.best_params_, summary


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> tuple:
    """
    RandomizedSearchCV on XGBoost.
    Returns (best_estimator, best_params, cv_summary_dict).
    """
    log.info("Tuning XGBoost with RandomizedSearchCV …")
    t0 = time.time()

    param_dist = {
        "n_estimators"   : cfg["tuning"]["xgboost_grid"]["n_estimators"],
        "learning_rate"  : cfg["tuning"]["xgboost_grid"]["learning_rate"],
        "max_depth"      : cfg["tuning"]["xgboost_grid"]["max_depth"],
        "subsample"      : cfg["tuning"]["xgboost_grid"]["subsample"],
        "colsample_bytree": cfg["tuning"]["xgboost_grid"]["colsample_bytree"],
        "reg_alpha"      : cfg["tuning"]["xgboost_grid"]["reg_alpha"],
        "reg_lambda"     : cfg["tuning"]["xgboost_grid"]["reg_lambda"],
    }

    base = xgb.XGBRegressor(
        random_state=cfg["models"]["random_state"],
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=cfg["tuning"]["n_iter"],
        cv=cfg["tuning"]["cv_folds"],
        scoring=cfg["tuning"]["scoring"],
        n_jobs=cfg["tuning"]["n_jobs"],
        random_state=cfg["models"]["random_state"],
        verbose=0,
    )
    search.fit(X_train, y_train)

    log.info(f"  Best params : {search.best_params_}")
    log.info(f"  Best CV R²  : {search.best_score_:.4f}  ({time.time()-t0:.1f}s)")

    kf = KFold(n_splits=cfg["tuning"]["cv_folds"], shuffle=True,
               random_state=cfg["models"]["random_state"])
    cv_res = cross_validate(search.best_estimator_, X_train, y_train,
                            cv=kf, scoring=SCORING, return_train_score=False)
    summary = _report_cv("Tuned XGBoost", cv_res)
    return search.best_estimator_, search.best_params_, summary


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cfg: dict, name: str) -> dict:
    """Run k-fold CV on any fitted or unfitted model. Returns summary dict."""
    kf = KFold(n_splits=cfg["tuning"]["cv_folds"], shuffle=True,
               random_state=cfg["models"]["random_state"])
    cv_res = cross_validate(model, X, y, cv=kf, scoring=SCORING,
                            return_train_score=True)
    summary = _report_cv(name, cv_res)
    summary["train_r2_mean"] = cv_res["train_r2"].mean()
    summary["train_r2_std"]  = cv_res["train_r2"].std()
    return summary
