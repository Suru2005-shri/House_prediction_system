"""
main.py  v2.0
=============
Upgraded House Price Prediction Pipeline

New in v2:
  YAML config | Logging | Hyperparameter tuning | k-Fold CV
  Stacking Ensemble | SHAP | Interactive Plotly | Learning curves
  Model persistence (scaler + feature_names) | Pro dashboard

Run:
    python main.py              # full pipeline
    python main.py --skip-tune  # skip hyperparameter tuning (faster)
    python main.py --skip-shap  # skip SHAP (if shap causes issues)
    streamlit run app.py        # launch the web app
"""

import os, sys, argparse, time
import yaml, joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))

from logger              import setup_logging, get_logger
from data_generator      import generate_housing_data
from preprocessor        import clean_data, feature_engineering, prepare_features
from models              import train_all_models, save_best_model
from tuner               import tune_random_forest, tune_xgboost
from stacking            import train_stacking_model
from explainer           import run_shap_analysis
from visualizer          import (
    plot_data_overview, plot_correlation_heatmap, plot_price_distribution,
    plot_feature_vs_price, plot_categorical_analysis, plot_model_comparison,
    plot_actual_vs_predicted, plot_residuals, plot_feature_importance,
    plot_price_trend_by_area,
)
from advanced_visualizer import (
    plot_interactive_price_map, plot_radar_comparison, plot_3d_price_surface,
    plot_learning_curves, plot_cv_score_distribution,
    plot_error_distribution, plot_pro_dashboard,
)
from predictor import build_input_vector, predict_price, print_prediction_report


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-tune",  action="store_true")
    p.add_argument("--skip-shap",  action="store_true")
    p.add_argument("--skip-stack", action="store_true")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

# ── Sample house ───────────────────────────────────────────────────────────────
SAMPLE_HOUSE = {
    "area_sqft": 2200, "bedrooms": 4, "bathrooms": 3, "floors": 2,
    "age_years": 8, "garage": 1, "garden": 1, "swimming_pool": 0,
    "location_score": 7.5, "distance_city": 6.0,
    "school_nearby": 1, "furnishing": 1,
}


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    setup_logging()
    log = get_logger("main")
    t0  = time.time()

    for d in [cfg["output"]["plots_dir"], cfg["output"]["models_dir"],
              cfg["output"]["logs_dir"], cfg["data"]["output_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Generate ────────────────────────────────────────────────────────────
    log.info("PHASE 1 | Dataset Generation")
    df_raw = generate_housing_data(cfg["data"]["n_samples"], cfg["data"]["random_state"])
    df_raw.to_csv(cfg["paths"]["raw_data"], index=False)
    log.info(f"  Saved {df_raw.shape} → {cfg['paths']['raw_data']}")

    # ── 2. Clean + engineer ────────────────────────────────────────────────────
    log.info("PHASE 2 | Cleaning & Feature Engineering")
    df_clean = clean_data(df_raw.copy())
    df_eng   = feature_engineering(df_clean.copy())
    df_eng.to_csv(cfg["paths"]["engineered_data"], index=False)
    log.info(f"  Engineered dataset {df_eng.shape} → {cfg['paths']['engineered_data']}")

    # ── 3. EDA ─────────────────────────────────────────────────────────────────
    log.info("PHASE 3 | EDA Visualizations")
    plot_data_overview(df_eng);         plot_correlation_heatmap(df_eng)
    plot_price_distribution(df_eng);    plot_feature_vs_price(df_eng)
    plot_categorical_analysis(df_eng);  plot_price_trend_by_area(df_eng)
    plot_interactive_price_map(df_eng); plot_3d_price_surface(df_eng)

    # ── 4. Prepare features ────────────────────────────────────────────────────
    log.info("PHASE 4 | Train-Test Split & Scaling")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features(
        df_eng, target=cfg["data"]["target_column"])
    joblib.dump(scaler,        cfg["paths"]["scaler"])
    joblib.dump(feature_names, cfg["paths"]["feature_names"])
    log.info(f"  Train {X_train.shape} | Test {X_test.shape}")

    # ── 5. Baseline models ─────────────────────────────────────────────────────
    log.info("PHASE 5 | Baseline Model Training")
    results = train_all_models(X_train, y_train, X_test, y_test)

    # ── 6. Hyperparameter tuning ───────────────────────────────────────────────
    cv_summaries = []
    if not args.skip_tune:
        log.info("PHASE 6 | Hyperparameter Tuning (RandomizedSearchCV)")
        tuned_rf,  _, rf_cv  = tune_random_forest(X_train, y_train, cfg)
        tuned_xgb, _, xgb_cv = tune_xgboost(X_train, y_train, cfg)
        cv_summaries = [rf_cv, xgb_cv]

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        for name, mdl in [("Tuned RF", tuned_rf), ("Tuned XGB", tuned_xgb)]:
            yp = mdl.predict(X_test)
            mae = mean_absolute_error(y_test, yp)
            rmse = np.sqrt(mean_squared_error(y_test, yp))
            r2   = r2_score(y_test, yp)
            results[name] = (mdl, yp, {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
            log.info(f"  {name} → R²={r2:.4f}  MAE=${mae:,.0f}")
    else:
        log.info("  Skipped (--skip-tune)")

    # ── 7. Stacking ensemble ───────────────────────────────────────────────────
    if not args.skip_stack:
        log.info("PHASE 7 | Stacking Ensemble")
        sm, sp, smets = train_stacking_model(X_train, y_train, X_test, y_test, cfg)
        results["Stacking Ensemble"] = (sm, sp, smets)
        joblib.dump(sm, cfg["paths"]["stacked_model"])
    else:
        log.info("  Skipped (--skip-stack)")

    # ── 8. Save best model ─────────────────────────────────────────────────────
    log.info("PHASE 8 | Save Best Model")
    best_name  = save_best_model(results, cfg["output"]["models_dir"])
    best_model = results[best_name][0]
    best_pred  = results[best_name][1]

    # ── 9. Evaluation plots ────────────────────────────────────────────────────
    log.info("PHASE 9 | Evaluation Visualizations")
    metrics_list     = [v[2] for v in results.values()]
    predictions_dict = {k: v[1] for k, v in results.items()}

    plot_model_comparison(metrics_list)
    plot_actual_vs_predicted(y_test, predictions_dict)
    plot_residuals(y_test, best_pred, best_name)
    plot_feature_importance(best_model, feature_names, best_name)
    plot_error_distribution(y_test, predictions_dict)
    plot_radar_comparison(metrics_list)
    if cv_summaries:
        plot_cv_score_distribution(cv_summaries)
    try:
        plot_learning_curves(best_model, X_train, y_train, best_name)
    except Exception as e:
        log.warning(f"  Learning curves skipped: {e}")
    plot_pro_dashboard(df_eng, metrics_list, y_test, best_pred,
                       best_name, feature_names, best_model)

    # ── 10. SHAP ───────────────────────────────────────────────────────────────
    if not args.skip_shap:
        log.info("PHASE 10 | SHAP Explainability")
        try:
            run_shap_analysis(best_model, X_train, X_test,
                              feature_names, best_name, cfg)
        except Exception as e:
            log.error(f"  SHAP failed: {e}  (try --skip-shap)")
    else:
        log.info("  Skipped (--skip-shap)")

    # ── 11. Predict sample ─────────────────────────────────────────────────────
    log.info("PHASE 11 | Sample Price Prediction")
    vec = build_input_vector(SAMPLE_HOUSE, scaler, feature_names)
    print_prediction_report(SAMPLE_HOUSE, predict_price(best_model, vec), best_name)

    # ── Final summary ──────────────────────────────────────────────────────────
    n_plots = len([f for f in os.listdir("outputs")
                   if f.endswith((".png", ".html"))])
    log.info("=" * 60)
    log.info("COMPLETE")
    log.info(f"  Best model : {best_name}")
    log.info(f"  Best R²    : {max(m['R2'] for m in metrics_list):.4f}")
    log.info(f"  Plots      : {n_plots} files in outputs/")
    log.info(f"  Time       : {time.time()-t0:.1f}s")
    log.info(f"  Web app    : streamlit run app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
