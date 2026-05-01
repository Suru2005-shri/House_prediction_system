"""
advanced_visualizer.py
----------------------
Interactive Plotly charts + enhanced Matplotlib plots.
All figures saved to outputs/ as PNG (static) and HTML (interactive).
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from logger import get_logger

log = get_logger(__name__)
sns.set_theme(style="darkgrid", palette="muted")
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def _savefig(fname: str):
    path = os.path.join(SAVE_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {path}")


def _savehtml(fig, fname: str):
    path = os.path.join(SAVE_DIR, fname)
    fig.write_html(path)
    log.info(f"  Saved → {path}")


# ── 1. Interactive price map (scatter by location score) ─────────────────────
def plot_interactive_price_map(df: pd.DataFrame):
    log.info("Plotting interactive price bubble chart …")
    fig = px.scatter(
        df.sample(min(800, len(df)), random_state=42),
        x="distance_city",
        y="price_usd",
        size="area_sqft",
        color="location_score",
        hover_data=["bedrooms", "bathrooms", "age_years", "furnishing"],
        color_continuous_scale="RdYlGn",
        size_max=20,
        title="House Price vs Distance from City  (bubble size = area, colour = location score)",
        labels={
            "distance_city": "Distance from City Centre (km)",
            "price_usd": "Price (USD)",
            "location_score": "Location Score",
        },
        template="plotly_dark",
    )
    fig.update_layout(font=dict(size=12), title_font_size=15)
    _savehtml(fig, "15_interactive_price_map.html")


# ── 2. Interactive model comparison radar chart ────────────────────────────────
def plot_radar_comparison(metrics_list: list):
    log.info("Plotting radar model comparison …")
    # Normalise each metric to 0-1 (R² already 0-1; MAE/RMSE inverted)
    df_m = pd.DataFrame(metrics_list)
    categories = ["R²", "1-MAE_norm", "1-RMSE_norm"]

    mae_max  = df_m["MAE"].max();  rmse_max = df_m["RMSE"].max()
    df_m["mae_n"]  = 1 - df_m["MAE"]  / mae_max
    df_m["rmse_n"] = 1 - df_m["RMSE"] / rmse_max

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    fig = go.Figure()
    for i, row in df_m.iterrows():
        vals = [row["R2"], row["mae_n"], row["rmse_n"]]
        vals += [vals[0]]   # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Model"],
            line=dict(color=colors[i % len(colors)]),
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Comparison — Radar Chart (all metrics 0→1, higher=better)",
        template="plotly_dark",
        font=dict(size=12),
        title_font_size=15,
    )
    _savehtml(fig, "16_radar_model_comparison.html")


# ── 3. Interactive 3-D price surface ─────────────────────────────────────────
def plot_3d_price_surface(df: pd.DataFrame):
    log.info("Plotting 3-D price surface …")
    sample = df.sample(min(600, len(df)), random_state=42)
    fig = px.scatter_3d(
        sample,
        x="area_sqft",
        y="location_score",
        z="price_usd",
        color="bedrooms",
        size="bathrooms",
        size_max=10,
        opacity=0.7,
        title="3-D Price Surface: Area × Location × Price",
        labels={
            "area_sqft":     "Area (sq ft)",
            "location_score":"Location Score",
            "price_usd":     "Price (USD)",
            "bedrooms":      "Bedrooms",
        },
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig.update_layout(title_font_size=15, font=dict(size=11))
    _savehtml(fig, "17_3d_price_surface.html")


# ── 4. Learning curves ────────────────────────────────────────────────────────
def plot_learning_curves(model, X_train: np.ndarray, y_train: np.ndarray,
                         model_name: str):
    """Training vs validation score at increasing training sizes."""
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import make_scorer, r2_score

    log.info(f"Plotting learning curves for {model_name} …")
    sizes = np.linspace(0.10, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=sizes,
        cv=5,
        scoring=make_scorer(r2_score),
        n_jobs=-1,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    vl_mean = val_scores.mean(axis=1)
    vl_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, tr_mean, "o-", color="#4C72B0", label="Training R²")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color="#4C72B0")
    ax.plot(train_sizes, vl_mean, "s-", color="#DD8452", label="Validation R²")
    ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std,
                    alpha=0.15, color="#DD8452")
    ax.set_xlabel("Training Samples", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title(f"Learning Curves — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    _savefig("18_learning_curves.png")


# ── 5. CV score distribution (violin/box) ────────────────────────────────────
def plot_cv_score_distribution(cv_summaries: list):
    """Violin plot of per-fold R² scores across models."""
    log.info("Plotting CV score distribution …")
    rows = []
    for s in cv_summaries:
        if "r2_scores" in s:
            for fold_score in s["r2_scores"]:
                rows.append({"Model": s["model"], "CV R²": fold_score})
    if not rows:
        log.warning("  No CV score data — skipping violin plot.")
        return

    df_cv = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_cv, x="Model", y="CV R²", ax=ax,
                   palette="Set2", inner="box", cut=0)
    ax.set_title("Cross-Validation R² Score Distribution per Model",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("R² Score")
    ax.axhline(df_cv["CV R²"].max(), color="red", linewidth=1,
               linestyle="--", label=f"Best: {df_cv['CV R²'].max():.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig("19_cv_score_distribution.png")


# ── 6. Prediction error distribution ─────────────────────────────────────────
def plot_error_distribution(y_test: np.ndarray, predictions_dict: dict):
    """KDE of percentage prediction error for each model."""
    log.info("Plotting error distribution …")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for (name, y_pred), color in zip(predictions_dict.items(), colors):
        pct_err = 100 * (y_pred - y_test) / y_test
        pct_err_clipped = np.clip(pct_err, -60, 60)
        ax.hist(pct_err_clipped, bins=50, alpha=0.55, label=name,
                color=color, density=True)

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero error")
    ax.set_xlabel("Prediction Error (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Prediction Error Distribution by Model",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig("20_error_distribution.png")


# ── 7. Final pro dashboard ────────────────────────────────────────────────────
def plot_pro_dashboard(df: pd.DataFrame, metrics_list: list,
                       y_test: np.ndarray, best_pred: np.ndarray,
                       best_name: str, feature_names: list, best_model):
    """A single 3×3 summary dashboard in dark GitHub style."""
    log.info("Plotting professional dashboard …")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

    BG = "#161b22"; TEXT = "#e6edf3"; GRID = "#21262d"
    COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657"]

    def dark(ax, title):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=TEXT, fontweight="bold", fontsize=10, pad=8)

    df_m   = pd.DataFrame(metrics_list)
    models = df_m["Model"].str.replace(" ", "\n").tolist()

    # (0,0) R² bar
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(models, df_m["R2"], color=COLORS[:len(df_m)], edgecolor=BG, width=0.6)
    ax.set_ylim(0, 1.08)
    for b, v in zip(bars, df_m["R2"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{v:.3f}", ha="center", fontsize=8, color=TEXT, fontweight="bold")
    dark(ax, "R² Score  (higher = better)")

    # (0,1) MAE bar
    ax = fig.add_subplot(gs[0, 1])
    bars2 = ax.bar(models, df_m["MAE"]/1000, color=COLORS[:len(df_m)], edgecolor=BG, width=0.6)
    for b, v in zip(bars2, df_m["MAE"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                f"${v/1000:.1f}K", ha="center", fontsize=8, color=TEXT, fontweight="bold")
    dark(ax, "MAE  ($K, lower = better)")

    # (0,2) Actual vs Predicted
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(y_test/1000, best_pred/1000, alpha=0.35, s=10, color="#58a6ff")
    lims = [min(y_test.min(), best_pred.min())/1000,
            max(y_test.max(), best_pred.max())/1000]
    ax.plot(lims, lims, "--", color="#f78166", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual ($K)"); ax.set_ylabel("Predicted ($K)")
    ax.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)
    dark(ax, f"Actual vs Predicted — {best_name.split()[0]}")

    # (1,0) Price distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(df["price_usd"]/1000, bins=40, color="#3fb950", edgecolor=BG, alpha=0.85)
    ax.set_xlabel("Price ($K)"); ax.set_ylabel("Count")
    dark(ax, "Price Distribution")

    # (1,1) Feature vs price (area)
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(df["area_sqft"], df["price_usd"]/1000,
                    c=df["location_score"], cmap="RdYlGn",
                    alpha=0.3, s=8)
    plt.colorbar(sc, ax=ax, label="Loc. Score").ax.yaxis.label.set_color(TEXT)
    ax.set_xlabel("Area (sq ft)"); ax.set_ylabel("Price ($K)")
    dark(ax, "Area vs Price (colour=location)")

    # (1,2) Residuals
    ax = fig.add_subplot(gs[1, 2])
    residuals = y_test - best_pred
    ax.scatter(best_pred/1000, residuals/1000, alpha=0.3, s=8, color="#d2a8ff")
    ax.axhline(0, color="#f78166", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted ($K)"); ax.set_ylabel("Residual ($K)")
    dark(ax, "Residual Plot")

    # (2,0) Correlation of top features with price
    ax = fig.add_subplot(gs[2, 0])
    num_df = df.select_dtypes(include=np.number)
    corrs = num_df.corr()["price_usd"].drop("price_usd").sort_values()
    colors_c = ["#f78166" if v < 0 else "#3fb950" for v in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors_c, edgecolor=BG)
    ax.axvline(0, color=TEXT, linewidth=0.8)
    dark(ax, "Feature Correlation with Price")

    # (2,1) Bedrooms box plot
    ax = fig.add_subplot(gs[2, 1])
    bd_groups = [df[df["bedrooms"]==bd]["price_usd"].values/1000
                 for bd in sorted(df["bedrooms"].unique())]
    bp = ax.boxplot(bd_groups, patch_artist=True, medianprops=dict(color=TEXT, linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(sorted(df["bedrooms"].unique()))
    ax.set_xlabel("Bedrooms"); ax.set_ylabel("Price ($K)")
    dark(ax, "Price by Bedroom Count")

    # (2,2) Feature importance (if tree model)
    ax = fig.add_subplot(gs[2, 2])
    if hasattr(best_model, "feature_importances_"):
        imps = best_model.feature_importances_
        idx = np.argsort(imps)[-12:]
        ax.barh(np.array(feature_names)[idx], imps[idx],
                color=plt.cm.viridis(np.linspace(0.2, 0.9, len(idx))),
                edgecolor=BG)
        ax.set_xlabel("Importance")
        dark(ax, f"Feature Importance — {best_name.split()[0]}")
    else:
        ax.text(0.5, 0.5, "Feature importance\nnot available\nfor this model",
                ha="center", va="center", color=TEXT, fontsize=10,
                transform=ax.transAxes)
        dark(ax, "Feature Importance")

    fig.suptitle("🏠  House Price Prediction  |  Project Dashboard  v2.0",
                 fontsize=17, color=TEXT, fontweight="bold", y=1.002)
    plt.savefig(os.path.join(SAVE_DIR, "00_pro_dashboard.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    log.info(f"  Saved → {os.path.join(SAVE_DIR, '00_pro_dashboard.png')}")
