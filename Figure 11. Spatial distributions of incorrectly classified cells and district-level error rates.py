# ============================ #
#   (0) Imports and basic settings
# ============================ #
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

fontname_tnr = "Times New Roman"

# ============================ #
#   (1) Paths and data
# ============================ #
save_path = r'YOUR_LOCAL_PATH\error_samples_and_district_distribution_merged'
os.makedirs(save_path, exist_ok=True)

file_path = r'YOUR_LOCAL_PATH\dataset_randomly generated beta version.xlsx'

# Administrative district names (for annotation)
district_names = [
    'Central Zhengzhou',
    'Xingyang & Shangjie',
    'Zhongmou',
    'Gongyi',
    'Xinmi',
    'Xinzheng',
    'Dengfeng'
]
district_id_to_name = {i: n for i, n in enumerate(district_names)}

# Ordered resilience categories (multi-class labels)
custom_order = ['R', 'L_F', 'L_M', 'L_S', 'M_F', 'M_M', 'M_S', 'H_F', 'H_M', 'H_S']

data = pd.read_excel(file_path)
data['district_id'] = data['administrative district ID']
data['district_name'] = data['district_id'].map(district_id_to_name)
data['numeric_label'] = pd.Categorical(
    data['combined category'],
    categories=custom_order,
    ordered=True
).codes

# Explanatory variables: drop identifiers / spatial coordinates / label columns
excluded_columns = [
    'Tid',
    'combined category',
    'row',
    'col',
    'numeric_label',
    'administrative district ID',
    'district_name',
    'loss category',
    'recovery category'
]
X = data.drop(columns=excluded_columns, errors='ignore')
y = data['numeric_label']

# ============================ #
#   (2) Training: follow “Task 1” algorithm
#       2.1 Base class weights
#       2.2 Spatial neighbourhood and boundary re-weighting
#       2.3 Dual-model training and selection by training accuracy
# ============================ #

# 2.1 Class-specific costs (used for base sample weights)
error_costs = {
    0: 1.3,
    1: 2.0,
    2: 1.6,
    3: 1.5,
    4: 1.5,
    5: 1.2,
    6: 1.6,
    7: 1.0,
    8: 1.1,
    9: 1.3
}
base_w = compute_sample_weight(class_weight=error_costs, y=y)

# 2.2 Spatial and boundary re-weighting
def get_neighborhood(df: pd.DataFrame, r: int, c: int) -> pd.DataFrame:
    """
    Return the 3×3 Moore neighbourhood of the grid cell centred at (r, c),
    including the focal cell itself.
    """
    return df[
        (df['row'] >= r - 1) & (df['row'] <= r + 1) &
        (df['col'] >= c - 1) & (df['col'] <= c + 1)
    ]


adjusted_w = []
boundary_threshold = 10
row_max = data['row'].max()
col_max = data['col'].max()

for idx, r in data.iterrows():
    neighbors = get_neighborhood(data, r['row'], r['col'])
    same_cls = (neighbors['numeric_label'] == r['numeric_label']).sum()
    w = base_w[idx]

    # Spatial-context-based re-weighting
    if same_cls == 2:
        w *= 1.1
    elif same_cls == 1:
        w *= 1.5
    elif same_cls == 0:
        # Isolated cell in its resilience category
        w *= 2.0
        # Additional increase for boundary cells
        if (
            r['col'] < boundary_threshold or
            r['col'] > (col_max - boundary_threshold) or
            r['row'] < boundary_threshold or
            r['row'] > (row_max - boundary_threshold)
        ):
            w *= 3.0

    adjusted_w.append(w)

# 2.3 Dual-model XGBoost training (same hyperparameters for both models)
params = dict(
    colsample_bytree=0.746,
    learning_rate=0.034,
    max_depth=8,
    n_estimators=270,
    subsample=0.822
)

# Model 1: base class weights
model1 = XGBClassifier(**params)
model1.fit(X, y, sample_weight=base_w)
pred1 = model1.predict(X)
acc1 = accuracy_score(y, pred1)

# Model 2: spatially re-weighted sample weights
model2 = XGBClassifier(**params)
model2.fit(X, y, sample_weight=np.array(adjusted_w))
pred2 = model2.predict(X)
acc2 = accuracy_score(y, pred2)

# Select the model with higher training-set accuracy (Task 1 rule)
best_pred = pred2 if acc2 > acc1 else pred1
data['predictions'] = best_pred

# ============================ #
#   (3) Error metrics and grid-level mapping
# ============================ #
group_by_district = data.groupby('district_id', dropna=False)

# District-level error rate (% of misclassified grids)
error_rate = group_by_district.apply(
    lambda g: (g['predictions'] != g['numeric_label']).mean() * 100.0
).to_dict()

data['error_rate'] = data['district_id'].map(lambda i: error_rate.get(i, 0.0))

# Absolute difference between predicted and true resilience-code
data['diff_category'] = (data['predictions'] - data['numeric_label']).abs()

# Grid-level maps (pivoted by row and col)
grid_error = data.pivot(index='row', columns='col', values='error_rate')

data['diff_category_mapped'] = 0
mask_incorrect = data['predictions'] != data['numeric_label']
data.loc[mask_incorrect, 'diff_category_mapped'] = data.loc[mask_incorrect, 'diff_category']

grid_diffcat = data.pivot(index='row', columns='col', values='diff_category_mapped')

# ============================ #
#   (4) Visualisation:
#       Task 2-style layout:
#       - Main map without ticks and frame, grayscale error rate,
#         coloured overlay for |prediction – label|
#       - Scale bar below the main map
#       - Subplot labels (a), (b), (c)
#       - Two pie charts on the right
# ============================ #
fig = plt.figure(figsize=(18, 10))

# Layout: main map on the left, two aligned pie charts on the right
map_left, map_bottom, map_width, map_height = 0.05, 0.12, 0.64, 0.78
pie_width, pie_height = 0.26, 0.34
pie_left = 0.70
pie1_bottom = map_bottom + map_height - pie_height
pie2_bottom = 0.12

ax_map = fig.add_axes([map_left, map_bottom, map_width, map_height])
ax_pie1 = fig.add_axes([pie_left, pie1_bottom, pie_width, pie_height])  # (b)
ax_pie2 = fig.add_axes([pie_left, pie2_bottom, pie_width, pie_height])  # (c)

# (a) Main map: grayscale error rate background + coloured diff-category overlay
cmap_gray = plt.cm.gray_r
norm_error = colors.Normalize(vmin=0, vmax=grid_error.max().max())

sns.heatmap(
    grid_error,
    cmap=cmap_gray,
    norm=norm_error,
    square=True,
    cbar=False,
    ax=ax_map,
    alpha=0.5
)

# Remove axis labels, ticks, and frame
ax_map.set_xlabel('')
ax_map.set_ylabel('')
ax_map.set_xticks([])
ax_map.set_yticks([])
ax_map.set_xlim(-0.5, grid_error.shape[1] - 0.5)
ax_map.set_ylim(-0.5, grid_error.shape[0] - 0.5)
ax_map.set_frame_on(False)
for spine in ax_map.spines.values():
    spine.set_visible(False)

# Colour map for |prediction – label| (categories 1–9)
diff_colors = [
    '#4C516D',
    '#5E7D7E',
    '#6D9B98',
    '#8CB78F',
    '#B4B486',
    '#D5A768',
    '#E3A857',
    '#DC8744',
    '#C95D63'
]

cmap_list = [(1, 1, 1, 0)] + [matplotlib.colors.to_rgba(c) for c in diff_colors]
cmap_diff = matplotlib.colors.ListedColormap(cmap_list)
norm_diff = colors.BoundaryNorm(np.arange(0, 11, 1), cmap_diff.N)

ax_map.imshow(
    grid_diffcat.values,
    cmap=cmap_diff,
    norm=norm_diff,
    interpolation='nearest',
    origin='upper',
    extent=[
        -0.5,
        grid_diffcat.shape[1] - 0.5,
        grid_diffcat.shape[0] - 0.5,
        -0.5
    ]
)

# District name + error rate label (white box with partial transparency)
for did, sub in data.groupby('district_id', dropna=False):
    if sub.empty:
        continue
    ax_map.text(
        sub['col'].mean(),
        sub['row'].mean(),
        f"{sub['district_name'].iloc[0]}\n"
        f"{(sub['predictions'] != sub['numeric_label']).mean() * 100.0:.2f}%",
        ha='center',
        va='center',
        fontsize=20,
        fontname=fontname_tnr,
        bbox=dict(
            facecolor='white',
            alpha=0.65,
            edgecolor='none',
            pad=2
        )
    )


def add_scalebar_below_left_map(
    fig_obj,
    map_ax,
    total_units=20,
    tick_interval=10,
    top_labels=("8 km", "16 km"),
    pad_frac=0.18,
    gap_fig=0.06,
    bar_h_fig=0.055,
    lw=2.6,
    fs=15,
    fontname="Times New Roman"
):
    """
    Add a horizontal scale bar below the main map, centred relative to the map width.
    The parameter total_units defines the map-unit length of the bar.
    """
    pos = map_ax.get_position()
    x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height
    xmin, xmax = map_ax.get_xlim()
    bar_length_fraction = float(total_units) / float(xmax - xmin)

    bar_w_fig = max(0.05, min(w * bar_length_fraction, w * 0.95))
    bar_x_fig = x0 + (w - bar_w_fig) / 2.0
    bar_y_fig = max(0.01, y0 - gap_fig)

    ax_bar = fig_obj.add_axes([bar_x_fig, bar_y_fig, bar_w_fig, bar_h_fig])
    ax_bar.set_xlim(0, total_units)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")

    # Main bar line
    ax_bar.plot(
        [0, total_units],
        [0.5, 0.5],
        color="black",
        lw=lw,
        solid_capstyle="butt"
    )

    pad = pad_frac
    # Tick marks at 0, tick_interval, and total_units
    for x in [0, tick_interval, total_units]:
        ax_bar.plot(
            [x, x],
            [0.5 - pad, 0.5 + pad],
            color="black",
            lw=lw
        )

    # Numeric labels below the bar
    ax_bar.text(
        0,
        0.5 - 2.3 * pad,
        "0",
        ha="center",
        va="top",
        fontsize=fs,
        fontname=fontname
    )
    ax_bar.text(
        tick_interval,
        0.5 - 2.3 * pad,
        f"{tick_interval}",
        ha="center",
        va="top",
        fontsize=fs,
        fontname=fontname
    )
    ax_bar.text(
        total_units,
        0.5 - 2.3 * pad,
        f"{total_units}",
        ha="center",
        va="top",
        fontsize=fs,
        fontname=fontname
    )

    # Optional labels above the bar (e.g., approximate distances in km)
    if top_labels and len(top_labels) == 2:
        ax_bar.text(
            tick_interval,
            0.5 + 2.0 * pad,
            top_labels[0],
            ha="center",
            va="bottom",
            fontsize=fs,
            fontname=fontname
        )
        ax_bar.text(
            total_units,
            0.5 + 2.0 * pad,
            top_labels[1],
            ha="center",
            va="bottom",
            fontsize=fs,
            fontname=fontname
        )


# Add scale bar below the main map
add_scalebar_below_left_map(
    fig,
    ax_map,
    total_units=20,
    tick_interval=10,
    top_labels=("8 km", "16 km"),
    pad_frac=0.18,
    lw=2.6,
    fs=20,
    fontname=fontname_tnr
)

# Subplot label for the main panel
ax_map.text(
    0.01,
    0.985,
    "(a)",
    transform=ax_map.transAxes,
    ha='left',
    va='top',
    fontsize=20,
    fontname=fontname_tnr
)

# (b) Upper-right pie chart: proportion of misclassified vs correctly classified grids
num_incorrect = mask_incorrect.sum()
num_total = len(data)
num_correct = num_total - num_incorrect

ax_pie1.pie(
    [num_incorrect, num_correct],
    colors=['#6D9B98', 'lightgray'],
    startangle=270,
    autopct=lambda p: (
        f"{p:.1f}% incorrect" if p > 50.0 else f"{p:.1f}% correct"
    ),
    textprops={'fontsize': 20, 'fontname': fontname_tnr},
    radius=0.97
)
ax_pie1.text(
    0.02,
    0.98,
    "(b)",
    transform=ax_pie1.transAxes,
    ha='left',
    va='top',
    fontsize=20,
    fontname=fontname_tnr
)

# (c) Lower-right pie chart: composition of misclassified samples by |prediction – label|
incorrect_data = data[mask_incorrect].copy()
diff_counts = incorrect_data['diff_category'].value_counts().sort_index()
diff_counts = diff_counts.reindex(range(1, 10), fill_value=0)
sizes = diff_counts.values
total = sizes.sum() if sizes.sum() > 0 else 1

wedges2, _ = ax_pie2.pie(
    sizes,
    colors=diff_colors,
    startangle=0,
    radius=1.08
)

ax_pie2.set_aspect('equal', adjustable='box')
ax_pie2.set_xlim(-1.25, 1.60)
ax_pie2.set_ylim(-1.25, 1.25)

# Angles of wedge centres
angles = [0.5 * (w.theta1 + w.theta2) for w in wedges2]
r_edge = 1.03
r_inner_text = 0.58
r_outer_text = 1.48
tangential_shift = 0.06
cat7_vertical_shift = 0.015

for i, wdg in enumerate(wedges2):
    cat = i + 1
    pct = sizes[i] / total * 100.0
    ang_rad = np.deg2rad(angles[i])
    x = np.cos(ang_rad)
    y = np.sin(ang_rad)

    line_x = r_edge * x
    line_y = r_edge * y

    # Categories 6–9: external labels with leader lines
    if cat in [6, 7, 8, 9]:
        outx = r_outer_text * x + tangential_shift * (-y)
        outy = r_outer_text * y + tangential_shift * x
        if cat == 7:
            outy += cat7_vertical_shift

        ha = 'left' if x >= 0 else 'right'
        va = 'bottom' if y >= 0 else 'top'

        ax_pie2.annotate(
            f"Cat {cat} {pct:.1f}%",
            xy=(line_x, line_y),
            xytext=(outx, outy),
            textcoords='data',
            ha=ha,
            va=va,
            fontsize=15,
            fontname=fontname_tnr,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-",
                color=wdg.get_facecolor(),
                lw=1.2,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.12" if x >= 0 else "arc3,rad=-0.12"
            )
        )
    else:
        # Categories 1–5: internal labels
        ax_pie2.text(
            r_inner_text * x,
            r_inner_text * y,
            f"Cat {cat}\n{pct:.1f}%",
            ha='center',
            va='center',
            fontsize=20,
            fontname=fontname_tnr
        )

ax_pie2.text(
    0.02,
    0.98,
    "(c)",
    transform=ax_pie2.transAxes,
    ha='left',
    va='top',
    fontsize=20,
    fontname=fontname_tnr
)

# Vertical arrow from panel (b) to panel (c)
ax_pie1.annotate(
    '',
    xy=(0.5, 1.02),
    xycoords=ax_pie2.transAxes,
    xytext=(0.5, -0.02),
    textcoords=ax_pie1.transAxes,
    arrowprops=dict(
        arrowstyle="->",
        color='black',
        lw=1.6
    )
)

# ============================ #
#   (5) Save figure
# ============================ #
plt.tight_layout()

out_path = os.path.join(save_path, 'Composite_Figure.png')
plt.savefig(out_path, dpi=300)
print(f"Figure saved to: {out_path}")
print(f"Training accuracy: base_w={acc1:.4f} | spatial_w={acc2:.4f} | "
      f"selected={'spatial' if acc2 > acc1 else 'base'}")

