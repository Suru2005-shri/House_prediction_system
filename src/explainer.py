"""
explainer.py
------------
SHAP (SHapley Additive exPlanations) model explainability.

Answers:
  • Which features matter most globally?
  • Why did the model predict THIS price for THIS house?
  • How does each feature's value push the price up or down?

Works with: Random Forest, XGBoost (TreeExplainer — fast & exact),
            Linear Regression (LinearExplainer).
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from logger import get_logger

log = get_logger(__name__)
SAVE_DIR = "outputs"


def _save(fname: str):
    path = os.path.join(SAVE_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {path}")


# ── Build explainer ───────────────────────────────────────────────────────────

def build_explainer(model, X_background: np.ndarray):
    """
    Choose the right SHAP explainer for the model type.
    TreeExplainer for tree-based models (fast exact SHAP).
    LinearExplainer for linear models.
    """
    model_type = type(model).__name__

    if model_type in ("RandomForestRegressor", "XGBRegressor",
                      "GradientBoostingRegressor", "DecisionTreeRegressor",
                      "StackingRegressor"):
        log.info(f"  Using TreeExplainer for {model_type}")
        try:
            return shap.TreeExplainer(model)
        except Exception:
            log.warning("  TreeExplainer failed; falling back to KernelExplainer")
            bg = shap.kmeans(X_background, 50)
            return shap.KernelExplainer(model.predict, bg)

    elif model_type == "LinearRegression":
        log.info(f"  Using LinearExplainer for {model_type}")
        return shap.LinearExplainer(model, X_background)

    else:
        log.warning(f"  Unknown model type {model_type}; using KernelExplainer (slow)")
        bg = shap.kmeans(X_background, 50)
        return shap.KernelExplainer(model.predict, bg)


def compute_shap_values(explainer, X_sample: np.ndarray) -> np.ndarray:
    """Return SHAP values array of shape (n_samples, n_features)."""
    sv = explainer(X_sample)
    # Handle both old-style arrays and new Explanation objects
    if hasattr(sv, "values"):
        vals = sv.values
    else:
        vals = sv
    if vals.ndim == 3:          # multi-output: take first output
        vals = vals[:, :, 0]
    return vals


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray, X_sample: np.ndarray,
                      feature_names: list, model_name: str):
    """
    Beeswarm summary plot — shows both importance AND direction of each feature.
    Red dots = high feature value, Blue = low. Right = increases price.
    """
    log.info("Plotting SHAP summary (beeswarm) …")
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, max_display=16,
        plot_size=(12, 7),
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold", pad=12)
    _save("11_shap_summary_beeswarm.png")


def plot_shap_bar(shap_values: np.ndarray, feature_names: list, model_name: str):
    """Global mean |SHAP| bar chart — pure feature importance."""
    log.info("Plotting SHAP bar importance …")
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(idx)))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(feature_names)[idx], mean_abs[idx],
            color=colors, edgecolor="white")
    ax.set_xlabel("mean |SHAP value|  (USD impact)", fontsize=11)
    ax.set_title(f"Global Feature Importance — {model_name}",
                 fontsize=13, fontweight="bold")
    for i, v in enumerate(mean_abs[idx]):
        ax.text(v + 200, i, f"${v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    _save("12_shap_bar_importance.png")


def plot_shap_waterfall(explainer, X_sample: np.ndarray,
                        feature_names: list, sample_idx: int = 0):
    """
    Waterfall plot for ONE prediction — shows exactly how each feature
    pushed the predicted price up or down from the baseline.
    """
    log.info(f"Plotting SHAP waterfall for sample index {sample_idx} …")
    sv = explainer(X_sample[sample_idx:sample_idx+1])
    fig, ax = plt.subplots(figsize=(11, 7))
    shap.plots.waterfall(sv[0], max_display=16, show=False)
    plt.title("SHAP Waterfall — Why This Price?", fontsize=13,
              fontweight="bold", pad=12)
    _save("13_shap_waterfall.png")


def plot_shap_dependence(shap_values: np.ndarray, X_sample: np.ndarray,
                         feature_names: list, top_n: int = 2):
    """
    Dependence plots for the top-N most important features.
    Shows how price changes as that feature varies.
    """
    log.info("Plotting SHAP dependence plots …")
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_abs)[::-1][:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(7 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    for ax, feat_idx in zip(axes, top_features):
        feat_name = feature_names[feat_idx]
        x_vals = X_sample[:, feat_idx]
        y_vals = shap_values[:, feat_idx]
        sc = ax.scatter(x_vals, y_vals, c=x_vals, cmap="RdYlGn",
                        alpha=0.6, s=18)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat_name, fontsize=11)
        ax.set_ylabel("SHAP value (price impact, USD)", fontsize=10)
        ax.set_title(f"Dependence: {feat_name}", fontweight="bold")
        plt.colorbar(sc, ax=ax, label=feat_name)

    plt.suptitle("SHAP Dependence Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("14_shap_dependence.png")


# ── Master SHAP runner ────────────────────────────────────────────────────────

def run_shap_analysis(model, X_train: np.ndarray, X_test: np.ndarray,
                      feature_names: list, model_name: str, cfg: dict):
    """
    Full SHAP pipeline:
      1. Build explainer
      2. Compute SHAP values on a sample of test data
      3. Generate all four SHAP plots
    """
    log.info(f"Running SHAP analysis for {model_name} …")
    n = min(cfg["shap"]["n_samples"], X_test.shape[0])
    rng = np.random.RandomState(cfg["data"]["random_state"])
    idx = rng.choice(X_test.shape[0], n, replace=False)
    X_sample = X_test[idx]

    explainer   = build_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X_sample)

    plot_shap_summary   (shap_values, X_sample, feature_names, model_name)
    plot_shap_bar       (shap_values, feature_names, model_name)
    plot_shap_waterfall (explainer, X_sample, feature_names, sample_idx=0)
    plot_shap_dependence(shap_values, X_sample, feature_names, top_n=2)

    log.info("SHAP analysis complete — 4 plots saved.")
    return shap_values, explainer
