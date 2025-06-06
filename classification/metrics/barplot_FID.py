import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("fid_per_class.csv")

# --- Filter classes with ≥1000 samples ---
df_filtered = df[df["n_samples"] >= 1000].sort_values("FID")

# --- Plot style ---
sns.set(style="whitegrid", context="paper", font="DejaVu Sans", font_scale=1.2)
palette = sns.color_palette("viridis", len(df_filtered))

# --- Create figure ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_filtered,
    x="Clase",
    y="FID",
    palette=palette
)

# --- Annotate FID values above bars ---
for i, row in enumerate(df_filtered.itertuples()):
    ax.text(
        i, 
        row.FID + 1.5,  # small offset above the bar
        f"{row.FID:.1f}", 
        ha='center', 
        va='bottom',
        fontsize=10
    )

# --- Axis formatting ---
ax.set_ylabel("FID score", fontsize=13)
ax.set_xlabel("Cell class", fontsize=13)
ax.set_title("Frechet Inception Distance (FID) per cell class", fontsize=14, pad=12)
ax.set_ylim(0, 110)  # Extend y-axis to avoid overlap
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# --- Save plot ---
output_base = "fid_barplot"
plt.savefig(f"{output_base}.png", dpi=300)
plt.savefig(f"{output_base}.pdf", bbox_inches='tight')

print("✅ Final professional plot saved as fid_barplot_final.[png/pdf]")
plt.show()