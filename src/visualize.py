"""
visualizer.py
-------------
All EDA charts, evaluation plots, and feature importance graphs.
Every figure is saved to the outputs/ folder for GitHub upload.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

def _save(fname: str):
    path = os.path.join(SAVE_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved → {path}")


# ── 1. Dataset overview ───────────────────────────────────────────────────────
def plot_data_overview(df: pd.DataFrame):
    """Distribution of every numeric column."""
    print("\n[Visualizer] Plotting data overview...")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    n = len(numeric_cols)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions – Housing Dataset", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("01_data_overview.png")


# ── 2. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation matrix as an annotated heatmap."""
    print("[Visualizer] Plotting correlation heatmap...")
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    _save("02_correlation_heatmap.png")


# ── 3. Price distribution ─────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame):
    """Histogram + KDE of house prices."""
    print("[Visualizer] Plotting price distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution
    sns.histplot(df["price_usd"], bins=40, kde=True, color="#2196F3", ax=axes[0])
    axes[0].set_title("House Price Distribution", fontweight="bold")
    axes[0].set_xlabel("Price (USD)")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))

    # Log-scale
    sns.histplot(np.log1p(df["price_usd"]), bins=40, kde=True, color="#FF5722", ax=axes[1])
    axes[1].set_title("Log-Transformed Price Distribution", fontweight="bold")
    axes[1].set_xlabel("log(Price + 1)")

    plt.suptitle("Target Variable Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("03_price_distribution.png")


# ── 4. Feature vs price scatter plots ────────────────────────────────────────
def plot_feature_vs_price(df: pd.DataFrame):
    """Key numeric features plotted against price."""
    print("[Visualizer] Plotting features vs price...")
    key_features = ["area_sqft", "location_score", "bedrooms", "age_years",
                    "distance_city", "bathrooms"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for i, feat in enumerate(key_features):
        axes[i].scatter(df[feat], df["price_usd"], alpha=0.3, s=12, color=colors[i])
        # Trend line
        m, b = np.polyfit(df[feat], df["price_usd"], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(x_line, m * x_line + b, color="red", linewidth=1.5, label="Trend")
        axes[i].set_xlabel(feat, fontsize=10)
        axes[i].set_ylabel("Price (USD)", fontsize=9)
        axes[i].set_title(f"{feat} vs Price", fontweight="bold")
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        axes[i].legend(fontsize=8)

    plt.suptitle("Feature vs House Price (Scatter Plots)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("04_feature_vs_price.png")


# ── 5. Box plots – categorical features ──────────────────────────────────────
def plot_categorical_analysis(df: pd.DataFrame):
    """Box plots for categorical/ordinal features vs price."""
    print("[Visualizer] Plotting categorical analysis...")
    cat_cols = ["bedrooms", "furnishing", "floors", "garage",
                "garden", "swimming_pool", "school_nearby"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    palette = sns.color_palette("Set2", 8)
    for i, col in enumerate(cat_cols):
        sns.boxplot(x=df[col], y=df["price_usd"], ax=axes[i],
                    palette=palette, width=0.5)
        axes[i].set_title(f"Price by {col}", fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Price (USD)")
        axes[i].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    axes[-1].set_visible(False)
    plt.suptitle("Price Distribution by Categorical Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("05_categorical_analysis.png")


# ── 6. Model comparison bar chart ─────────────────────────────────────────────
def plot_model_comparison(metrics_list: list):
    """Side-by-side bar charts for MAE, RMSE, R² across models."""
    print("[Visualizer] Plotting model comparison...")
    df_m = pd.DataFrame(metrics_list)
    models = df_m["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # MAE
    bars1 = axes[0].bar(x, df_m["MAE"], width=0.5, color=colors, edgecolor="white")
    axes[0].set_title("Mean Absolute Error (lower = better)", fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("MAE (USD)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"${bar.get_height()/1e3:.1f}K", ha="center", va="bottom", fontsize=9)

    # RMSE
    bars2 = axes[1].bar(x, df_m["RMSE"], width=0.5, color=colors, edgecolor="white")
    axes[1].set_title("Root Mean Squared Error (lower = better)", fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylabel("RMSE (USD)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"${bar.get_height()/1e3:.1f}K", ha="center", va="bottom", fontsize=9)

    # R²
    bars3 = axes[2].bar(x, df_m["R2"], width=0.5, color=colors, edgecolor="white")
    axes[2].set_title("R² Score (higher = better)", fontweight="bold")
    axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=15, ha="right")
    axes[2].set_ylabel("R² Score")
    axes[2].set_ylim(0, 1.05)
    for bar in bars3:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("06_model_comparison.png")


# ── 7. Actual vs Predicted ────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, predictions_dict: dict):
    """Scatter: actual price vs predicted price for each model."""
    print("[Visualizer] Plotting actual vs predicted...")
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.35, s=15, color=color, label="Predictions")
        # Perfect prediction line
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Price (USD)", fontsize=10)
        ax.set_ylabel("Predicted Price (USD)", fontsize=10)
        ax.set_title(name, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax.legend(fontsize=9)

    plt.suptitle("Actual vs Predicted House Prices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("07_actual_vs_predicted.png")


# ── 8. Residual plot ──────────────────────────────────────────────────────────
def plot_residuals(y_test, best_pred, best_name: str):
    """Residual analysis for the best model."""
    print("[Visualizer] Plotting residuals...")
    residuals = y_test - best_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(best_pred, residuals, alpha=0.35, s=12, color="#4C72B0")
    axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Price (USD)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title(f"{best_name} – Residuals vs Predicted", fontweight="bold")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    # Residual distribution
    axes[1].hist(residuals, bins=40, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_title("Residual Distribution", fontweight="bold")

    plt.suptitle("Residual Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("08_residual_analysis.png")


# ── 9. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str):
    """Horizontal bar chart of feature importances (tree models only)."""
    if not hasattr(model, "feature_importances_"):
        print(f"  [Visualizer] {model_name} has no feature_importances_ – skipping.")
        return

    print("[Visualizer] Plotting feature importance...")
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(idx)))
    ax.barh(np.array(feature_names)[idx], importances[idx],
            color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Feature Importance – {model_name}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)

    # Value labels
    for i, v in enumerate(importances[idx]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    _save("09_feature_importance.png")


# ── 10. Price trend by area ───────────────────────────────────────────────────
def plot_price_trend_by_area(df: pd.DataFrame):
    """Line/scatter showing how price scales with area, split by bedroom count."""
    print("[Visualizer] Plotting price trend by area...")
    fig, ax = plt.subplots(figsize=(12, 7))

    palette = sns.color_palette("tab10", 6)
    for bd, grp in df.groupby("bedrooms"):
        grp_sorted = grp.sort_values("area_sqft")
        ax.scatter(grp_sorted["area_sqft"], grp_sorted["price_usd"],
                   label=f"{bd} BR", alpha=0.3, s=12, color=palette[int(bd)-1])
        # Smooth trend line (rolling mean)
        if len(grp_sorted) > 20:
            rm = grp_sorted.set_index("area_sqft")["price_usd"].rolling(50, min_periods=5).mean()
            ax.plot(rm.index, rm.values, linewidth=2, color=palette[int(bd)-1])

    ax.set_xlabel("Area (sq ft)", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.set_title("House Price Trend by Area & Bedrooms", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.legend(title="Bedrooms", fontsize=9)
    plt.tight_layout()
    _save("10_price_trend_by_area.png")
