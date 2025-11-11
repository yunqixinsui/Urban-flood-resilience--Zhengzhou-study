# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe

# -----------------------------
# 1. Global font and size settings for matplotlib
# -----------------------------
matplotlib.rcParams['font.family'] = 'Times New Roman'  # Use Times New Roman for all text
matplotlib.rcParams['font.size'] = 32                   # Base font size
matplotlib.rcParams['axes.labelsize'] = 32              # Axis label font size
matplotlib.rcParams['axes.titlesize'] = 32              # Figure title font size
matplotlib.rcParams['xtick.labelsize'] = 32             # X-axis tick label font size
matplotlib.rcParams['ytick.labelsize'] = 32             # Y-axis tick label font size
matplotlib.rcParams['legend.fontsize'] = 32             # Legend font size

# -----------------------------
# 2. Read the Excel file
# -----------------------------
file_path = r'YOUR_LOCAL_PATH\original_OD.xlsx'  # Replace with the full path to your local Excel file
df = pd.read_excel(file_path)

# -----------------------------
# 3. Compute 2D distance to the administrative center grid of each district
# -----------------------------
center_coords = {
    0: (62, 118),  # Central Zhengzhou City
    1: (66, 82),   # Xingyang City and Shangjie District
    2: (58, 158),  # Zhongmou County
    3: (62, 33),   # Gongyi City
    4: (35, 81),   # Xinmi City
    5: (17, 127),  # Xinzheng City
    6: (25, 39)    # Dengfeng City
}

# Map each grid cell to the row and column of the corresponding district center
df['center_row'] = df['administrative district ID'].map(lambda x: center_coords[x][0])
df['center_col'] = df['administrative district ID'].map(lambda x: center_coords[x][1])

# Euclidean distance (in grid units) from each cell to the corresponding district center
df['distance_to_center_2d'] = np.sqrt(
    (df['row'] - df['center_row']) ** 2 +
    (df['col'] - df['center_col']) ** 2
)

# Z-score standardization of distance within each district
df['distance_to_center_2d_z'] = df.groupby('administrative district ID')['distance_to_center_2d'] \
                                  .transform(lambda x: (x - x.mean()) / x.std())

# -----------------------------
# 4. Color palette and district names
# -----------------------------
colors = [
    '#6D9B98',  # ID = 0: Central Zhengzhou City
    '#D5A768',  # ID = 1: Xingyang City and Shangjie District
    '#4C516D',  # ID = 2: Zhongmou County
    '#E3A857',  # ID = 3: Gongyi City
    '#8CB78F',  # ID = 4: Xinmi City
    '#547AA5',  # ID = 5: Xinzheng City
    '#B4B486',  # ID = 6: Dengfeng City
]

district_names = {
    0: "Central Zhengzhou",
    1: "Xingyang & Shangjie",
    2: "Zhongmou",
    3: "Gongyi",
    4: "Xinmi",
    5: "Xinzheng",
    6: "Dengfeng"
}

# Automatically insert a line break after '&' to improve layout at large font sizes
district_names_wrapped = {
    k: re.sub(r'\s*&\s*', ' &\n', v) for k, v in district_names.items()
}

# -----------------------------
# 5. Construct pivot table and plot the grid
# -----------------------------
grid_data = df.pivot(index='row', columns='col', values='administrative district ID')

plt.figure(figsize=(20, 10))
sns.heatmap(
    grid_data,
    cmap=sns.color_palette(colors),
    cbar=False,
    square=True
)

# -----------------------------
# 6. Annotate administrative centers (hollow X) and district names
# -----------------------------
LABEL_FS = 26
PATH_EFFECTS = [pe.withStroke(linewidth=3, foreground="white")]  # White stroke to improve legibility

for district_id, (c_row, c_col) in center_coords.items():
    # Hollow X marker at the administrative center of each district
    plt.scatter(
        c_col, c_row,
        marker='X',
        s=120,
        facecolors='none',   # Hollow marker
        edgecolors='black',  # Edge color of the marker
        linewidths=1.5,
        label='_nolegend_'   # Exclude these markers from the automatic legend
    )
    # District name label near the administrative center (with line wrapping if needed)
    plt.text(
        c_col, c_row + 2,
        district_names_wrapped[district_id],
        color='black',
        fontsize=LABEL_FS,
        ha='center',
        va='bottom',
        multialignment='center',
        linespacing=1.15,
        path_effects=PATH_EFFECTS,
        clip_on=False
    )

# -----------------------------
# 7. Add a legend entry for the hollow X marker
# -----------------------------
center_handle = plt.scatter(
    [], [],  # Dummy points for the legend handle
    marker='X',
    s=120,
    facecolors='none',
    edgecolors='black',
    linewidths=1.5,
    label='District administrative center'
)

plt.legend(
    handles=[center_handle],
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    borderaxespad=0.0
)

# -----------------------------
# 8. Axis labels, ticks, and title
# -----------------------------
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Administrative Districts with Center Grids')

plt.xticks(rotation=0)
plt.yticks(rotation=0)

x_ticks = np.arange(0, 201, 10)
y_ticks = np.arange(0, 101, 10)
plt.xticks(ticks=x_ticks, labels=x_ticks)
plt.yticks(ticks=y_ticks, labels=y_ticks)

plt.xlim(-0.5, 200 - 0.5)
plt.ylim(-0.5, 100 - 0.5)

# Add an outer border to the axes
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

plt.tight_layout()

# -----------------------------
# 9. Save figure and updated dataset
# -----------------------------
output_fig_path = r'YOUR_LOCAL_PATH\districts_2d_centers.png'  # Replace with the full path to your local output figure
plt.savefig(output_fig_path, dpi=400, bbox_inches='tight')
plt.close()

output_data_path = r'YOUR_LOCAL_PATH\updated_dataset_with_2d_distances.xlsx'  # Replace with the full path to your local output Excel file
df.to_excel(output_data_path, index=False)
