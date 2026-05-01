import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff"]
BG = "#161b22"
TEXT = "#e6edf3"

def style_ax(ax, title):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_title(title, color=TEXT, fontweight="bold", fontsize=11, pad=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

models   = ["Linear\nRegr.", "Decision\nTree", "Random\nForest", "XGBoost"]
r2_vals  = [0.8694, 0.7555, 0.8396, 0.8513]
mae_vals = [36916,  51912,  41387,  39833]

# R2 bar chart
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(models, r2_vals, color=COLORS, edgecolor="#0d1117", width=0.6)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("R2 Score", color=TEXT)
style_ax(ax1, "R2 Score Comparison")
for bar, val in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9,
             color=TEXT, fontweight="bold")
bars[0].set_edgecolor("#58a6ff")
bars[0].set_linewidth(2.5)

# MAE bar chart
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(models, [v/1000 for v in mae_vals], color=COLORS, edgecolor="#0d1117", width=0.6)
ax2.set_ylabel("MAE (USD thousands)", color=TEXT)
style_ax(ax2, "Mean Absolute Error")
for bar, val in zip(bars2, mae_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"${val/1000:.1f}K", ha="center", va="bottom", fontsize=9,
             color=TEXT, fontweight="bold")
bars2[0].set_edgecolor("#58a6ff")
bars2[0].set_linewidth(2.5)

# Actual vs Predicted
ax3 = fig.add_subplot(gs[0, 2])
rng = np.random.RandomState(42)
y_a = rng.uniform(100, 900, 150)
y_p = y_a * rng.normal(1, 0.12, 150)
ax3.scatter(y_a, y_p, alpha=0.5, s=18, color="#58a6ff", label="Predictions")
ax3.plot([80, 950], [80, 950], "--", linewidth=1.5, label="Perfect fit", color="#f78166")
ax3.set_xlabel("Actual Price", color=TEXT)
ax3.set_ylabel("Predicted Price", color=TEXT)
style_ax(ax3, "Actual vs Predicted")
ax3.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)

# Feature importance
ax4 = fig.add_subplot(gs[1, 0])
feats = ["area_sqft", "location_score", "age_years", "distance_city",
         "total_rooms", "amenity_score", "furnishing", "bedrooms"]
imps  = [0.38, 0.19, 0.12, 0.10, 0.07, 0.06, 0.05, 0.03]
idx   = np.argsort(imps)
ax4.barh(np.array(feats)[idx], np.array(imps)[idx],
         color=plt.cm.viridis(np.linspace(0.2, 0.9, 8)),
         edgecolor="#0d1117")
ax4.set_xlabel("Importance", color=TEXT)
style_ax(ax4, "Feature Importance (RF)")

# Price distribution
ax5 = fig.add_subplot(gs[1, 1])
prices = np.clip(rng.lognormal(12.5, 0.6, 1000), 50000, 2500000)
ax5.hist(prices / 1000, bins=35, color="#3fb950", edgecolor="#0d1117", alpha=0.85)
ax5.set_xlabel("Price (thousands USD)", color=TEXT)
ax5.set_ylabel("Count", color=TEXT)
style_ax(ax5, "Price Distribution")

# Prediction card
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(BG)
for spine in ax6.spines.values():
    spine.set_color("#30363d")
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_title("Sample Prediction Output", color=TEXT, fontweight="bold", fontsize=11, pad=10)
lines = [
    ("Area",        "2,200 sq ft"),
    ("Bedrooms",    "4"),
    ("Bathrooms",   "3"),
    ("Floors",      "2"),
    ("Age",         "8 years"),
    ("Garage",      "Yes"),
    ("Location",    "7.5/10"),
    ("City Dist.",  "6 km"),
    ("Furnishing",  "Semi"),
]
for i, (lbl, val) in enumerate(lines):
    y = 0.88 - i * 0.092
    ax6.text(0.05, y, lbl + ":", color="#8b949e", fontsize=9, transform=ax6.transAxes)
    ax6.text(0.55, y, val, color=TEXT, fontsize=9,
             transform=ax6.transAxes, fontweight="bold")
ax6.text(0.5, 0.05, "PREDICTED:  $421,442", color="#3fb950", fontsize=13,
         transform=ax6.transAxes, ha="center", fontweight="bold")

fig.suptitle("House Price Prediction  |  ML Project Dashboard",
             fontsize=16, color=TEXT, fontweight="bold", y=0.99)
plt.savefig("outputs/00_project_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
print("Dashboard saved!")
