import matplotlib
# You may change the backend (e.g., to 'Qt5Agg') according to the local environment.
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['font.family'] = 'Times New Roman'

# ---------- Data loading ----------
file_path = r'YOUR_LOCAL_PATH\dataset_randomly generated beta version.xlsx'
data = pd.read_excel(file_path)

# ---------- Original administrative district name mapping (used for subsequent simplification) ----------
code_to_city = {
    0: 'Center Zhengzhou City',       # C
    1: 'Xingyang City & Shangjie',    # F
    2: 'Zhongmou City',               # A
    3: 'Gongyi City',                 # G
    4: 'Xinmi City',                  # D
    5: 'Xinzheng City',               # B
    6: 'Dengfeng City'                # E
}

# ---------- Name simplification: remove administrative suffixes and convert "Center" to "Centre" (British English) ----------
def simplify_name(name: str) -> str:
    """
    Simplify the administrative district name by:
    (1) Removing 'City', 'County', and 'District';
    (2) Collapsing double spaces;
    (3) Replacing 'Center ' with 'Centre ' to adopt British spelling.
    """
    name = name.replace('City', '').replace('County', '').replace('District', '')
    name = name.replace('  ', ' ').replace(' & ', ' & ').strip()
    name = name.replace('Center ', 'Centre ')
    return name

code_to_simple = {k: simplify_name(v) for k, v in code_to_city.items()}

# ---------- Region labels A–G ----------
code_to_label = {0: 'C', 1: 'F', 2: 'A', 3: 'G', 4: 'D', 5: 'B', 6: 'E'}
label_to_id = {v: k for k, v in code_to_label.items()}

# ---------- Colour palette (mapped to A–G via id_to_color_index) ----------
colors = [
    '#4C516D',  # A
    '#547AA5',  # B
    '#6D7D98',  # C
    '#8CB78F',  # D
    '#B4B486',  # E
    '#D5A768',  # F
    '#E3A857',  # G
]

id_to_color_index = {
    2: 0,  # A
    5: 1,  # B
    0: 2,  # C
    4: 3,  # D
    6: 4,  # E
    1: 5,  # F
    3: 6,  # G
}

# ---------- Administrative centres (row, col, z) ----------
admin_centers = {
    6: [39, 25, 1.44],   # E
    2: [58, 158, 0.92],  # A
    3: [62, 33, 1.43],   # G
    1: [66, 82, 1.27],   # F
    4: [35, 81, 1.37],   # D
    5: [17, 127, 1.35],  # B
    0: [62, 118, 1.48],  # C
}

# Column name for the standardized mobile signalling value
z_col = 'mobile signaling in referential day'

# Filter out extreme values for improved visualisation (z <= 1.5)
data_filtered = data[data[z_col] <= 1.5].copy()

# ---------- Figure and axes configuration ----------
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

# (A) Scatter plot of all grid points with colour-coded administrative districts
for district_id in sorted(data_filtered['administrative district ID'].unique()):
    subset = data_filtered[data_filtered['administrative district ID'] == district_id]
    c = colors[id_to_color_index[district_id]]
    ax.scatter(
        subset['col'],
        subset['row'],
        subset[z_col],
        c=c,
        s=20,
        alpha=0.2
    )

# (B) Administrative centres: emphasised markers and textual labels (A–G only)
for district_id, (row_val, col_val, z_val) in admin_centers.items():
    c = colors[id_to_color_index[district_id]]
    label = code_to_label[district_id]
    ax.scatter(
        col_val,
        row_val,
        z_val,
        c=c,
        s=220,
        marker='X',
        edgecolor='k',
        linewidths=1.5,
        zorder=5
    )
    ax.text(
        col_val,
        row_val,
        z_val + 0.05,
        f"{label}",
        fontsize=28,          # Enlarged label size
        fontweight='bold',
        ha='center',
        va='bottom'
    )

# Axis labels and title (can be modified or removed if necessary)
ax.set_xlabel('Col (x)', fontsize=20, labelpad=10)
ax.set_ylabel('Row (y)', fontsize=20, labelpad=10)
ax.set_zlabel('Standardized Mobile Signaling (z)', fontsize=20, labelpad=10)
ax.set_title('3D Visualization of Administrative Regions and Centres', fontsize=24, pad=20)

# Axis limits and tick configuration
ax.set_zlim(-0.5, 1.5)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', labelsize=16)

# (C) Legend without frame, using simplified names and British spelling "centre"
label_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
legend_handles, legend_labels = [], []

for label in label_order:
    dist_id = label_to_id[label]
    c = colors[id_to_color_index[dist_id]]
    city_name_simple = code_to_simple[dist_id]
    handle = mlines.Line2D(
        [],
        [],
        marker='o',
        color=c,
        markerfacecolor=c,
        markersize=12,
        linewidth=0
    )
    legend_handles.append(handle)
    legend_labels.append(f"{label}: {city_name_simple}")

leg = ax.legend(
    legend_handles,
    legend_labels,
    loc='upper left',
    bbox_to_anchor=(1.05, 1.0),
    fontsize=16,
    title='District centre',
    title_fontsize=20,
    frameon=False
)

# Adjust the right margin to prevent the legend from being clipped
fig.subplots_adjust(left=0.10, right=0.82, top=0.90, bottom=0.10)

# ---------- Output path and saving ----------
out_png = r'YOUR_LOCAL_PATH\Figure_3D_centre.png'
plt.savefig(out_png, dpi=300)
plt.show()
print(f"Saved: {out_png}")
