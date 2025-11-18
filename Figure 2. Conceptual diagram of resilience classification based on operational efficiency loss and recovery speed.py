import matplotlib
# ------------------------------------------------------------
# Explicitly use the 'Agg' backend to avoid 'tostring_rgb' errors
# in headless or remote environments.
# ------------------------------------------------------------
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Global font configuration
# ------------------------------------------------------------
# Set the global font to Times New Roman for all text elements.
plt.rcParams['font.family'] = 'Times New Roman'

# ------------------------------------------------------------
# Output directory and file name
# ------------------------------------------------------------
save_dir = r"YOUR_LOCAL_PATH\resilience_9grid"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "resilience_9grid2505.png")

# ------------------------------------------------------------
# 1. Create figure and axes
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# ------------------------------------------------------------
# 2. Set coordinate limits for a 3×3 conceptual grid
# ------------------------------------------------------------
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

# ------------------------------------------------------------
# 3. Draw vertical and horizontal grid lines
#    (4 horizontal + 4 vertical lines create a 3×3 lattice)
# ------------------------------------------------------------
for i in range(4):
    ax.axhline(i, color='black', linewidth=2)
    ax.axvline(i, color='black', linewidth=2)

# ------------------------------------------------------------
# 4. Configure axis ticks and labels (academic English)
# ------------------------------------------------------------
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_xticklabels(
    ["Fast (<1 day)", "Medium (2–7 days)", "Slow (>7 days)"],
    fontsize=20
)

ax.set_yticks([2.5, 1.5, 0.5])
ax.set_yticklabels(
    ["Low (0.1–0.3)", "Medium (0.3–0.6)", "High (>0.6)"],
    fontsize=20
)

# ------------------------------------------------------------
# 5. Axis titles (OERT vs OELR)
# ------------------------------------------------------------
ax.set_xlabel("Recovery Time (OERT)", fontsize=20, labelpad=10)
ax.set_ylabel("Urban Operational Efficiency Loss Rate (OELR)", fontsize=20, labelpad=10)

# ------------------------------------------------------------
# 6. Define labels and colours for the 3×3 resilience categories
#    Rows: OELR (from low to high)
#    Columns: OERT (from fast to slow)
# ------------------------------------------------------------
labels = [
    ["L_F", "L_M", "L_S"],  # Top row (low OELR)
    ["M_F", "M_M", "M_S"],  # Middle row (medium OELR)
    ["H_F", "H_M", "H_S"]   # Bottom row (high OELR)
]

colors = [
    ["#547AA5", "#5E7D7E", "#6D9B98"],  # L_F, L_M, L_S
    ["#8CB78F", "#B4B486", "#D5A768"],  # M_F, M_M, M_S
    ["#E3A857", "#DC8744", "#C95D63"]   # H_F, H_M, H_S
]

# ------------------------------------------------------------
# 7. Fill each cell with the assigned colour and add the label at the centre
# ------------------------------------------------------------
for row in range(3):
    for col in range(3):
        # Background rectangle for the current resilience category
        rect = plt.Rectangle(
            (col, 2 - row),   # lower-left corner in (x, y)
            1,                # width
            1,                # height
            facecolor=colors[row][col],
            alpha=0.5,
            edgecolor="none"
        )
        ax.add_patch(rect)

        # Text label at the centre of the cell
        x = col + 0.5
        y = 2.5 - row
        ax.text(
            x,
            y,
            labels[row][col],
            ha='center',
            va='center',
            fontsize=24,
            fontweight='bold'
        )

# ------------------------------------------------------------
# 8. Final layout: equal aspect ratio and academic-style title
# ------------------------------------------------------------
ax.set_aspect('equal')
ax.set_title("Resilience Category Matrix (OERT–OELR)", fontsize=20, pad=15)

# ------------------------------------------------------------
# 9. Save figure
# ------------------------------------------------------------
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved to: {save_path}")
