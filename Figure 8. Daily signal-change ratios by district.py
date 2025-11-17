import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib as mpl

# ------------------------------------------------------------
# 1. Global font configuration (Times New Roman, enlarged sizes)
# ------------------------------------------------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.labelsize'] = 30       # Axis label size
mpl.rcParams['axes.titlesize'] = 32       # Figure title size
mpl.rcParams['xtick.labelsize'] = 26      # x-axis tick label size
mpl.rcParams['ytick.labelsize'] = 26      # y-axis tick label size

# ------------------------------------------------------------
# 2. Input data
# ------------------------------------------------------------
file_path = r'YOUR_LOCAL_PATH\original OD_randomly generated beta version.xlsx'
df = pd.read_excel(file_path)

print("Columns in the DataFrame:", df.columns.tolist())

# Column that stores administrative district names
district_column = 'District/day'

# Desired plotting order of administrative districts
desired_order = [
    'Zhengzhou City',
    'Xinmi City',
    'Dengfeng City',
    'Zhongmou City',
    'Xinzheng City',
    'Central Zhengzhou City',
    'Xingyang City & Shangjie District',
    'Gongyi City'
]

# Keep only those districts that actually appear in the data
district_names = [
    district for district in desired_order
    if district in df[district_column].unique()
]

# Report districts that are missing in the dataset
missing_districts = [
    district for district in desired_order
    if district not in df[district_column].unique()
]
if missing_districts:
    print(
        "Warning: the following districts are missing in the data "
        "and will be ignored:",
        missing_districts
    )
else:
    print("All desired districts are present in the data.")

# Reorder the DataFrame to match the desired district order
df_ordered = df[df[district_column].isin(district_names)].copy()

df_ordered[district_column] = pd.Categorical(
    df_ordered[district_column],
    categories=district_names,
    ordered=True
)
df_ordered = df_ordered.sort_values(district_column)

district_names = df_ordered[district_column].tolist()
print("Ordered districts:", district_names)

# ------------------------------------------------------------
# 3. Date configuration
# ------------------------------------------------------------
# Dates (e.g. 719 = July 19), treated as column names
dates = [719, 720, 721, 726, 727, 728]

# Labels for days since rainfall start (e.g. 0, 1, 2, 7, 8, 9 days)
date_labels = ['0', '1', '2', '7', '8', '9']

# Check that all date columns exist in the DataFrame
missing_dates = [d for d in dates if d not in df_ordered.columns]
if missing_dates:
    print(
        "Warning: the following date columns are missing in the DataFrame:",
        missing_dates
    )

# ------------------------------------------------------------
# 4. Bar positions and colour mapping
# ------------------------------------------------------------
bar_width = 0.1
indices = np.arange(len(district_names))


def get_bar_color(value: float) -> str:
    """
    Map a numeric value to one of four discrete colours:
    (1) value ≤ 0.1
    (2) 0.1 < value ≤ 0.3
    (3) 0.3 < value ≤ 0.6
    (4) value > 0.6
    """
    if value <= 0.1:
        return '#547AA5'   # light blue
    elif 0.1 < value <= 0.3:
        return '#B4B486'   # low-saturation green
    elif 0.3 < value <= 0.6:
        return '#E3A857'   # yellow
    else:
        return '#DC8744'   # light orange


# Horizontal offsets for each date (creating a gap between 721 and 726)
offsets = {
    719: -2.5 * bar_width,   # 19 July
    720: -1.5 * bar_width,   # 20 July
    721: -0.5 * bar_width,   # 21 July
    # Deliberate gap of one bar_width between day 21 and day 26
    726: 1.5 * bar_width,    # 26 July
    727: 2.5 * bar_width,    # 27 July
    728: 3.5 * bar_width     # 28 July
}

# ------------------------------------------------------------
# 5. Plotting grouped bar chart
# ------------------------------------------------------------
plt.figure(figsize=(24, 12))

for date, label in zip(dates, date_labels):
    if date not in df_ordered.columns:
        # Skip dates that are missing
        continue

    offset = offsets.get(date, 0.0)
    bar_positions = indices + offset
    values = df_ordered[date].values

    color_list = [get_bar_color(v) for v in values]

    bars = plt.bar(
        bar_positions,
        values,
        width=bar_width,
        color=color_list,
        edgecolor='black'
    )

    # Add numeric labels above (or below) each bar
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height >= 0:
            plt.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=22,
                weight='bold'
            )
        else:
            plt.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -18),
                textcoords="offset points",
                ha='center',
                va='top',
                fontsize=22,
                weight='bold'
            )

    # Place day-since-rainfall labels below the x-axis
    for pos in bar_positions:
        plt.text(
            pos,
            -0.05,
            label,
            ha='center',
            va='top',
            fontsize=24,
            rotation=0
        )

# Remove default x tick labels (we only show day labels under bars)
plt.xticks(indices, [], rotation=0)

# Axis labels
plt.ylabel('OELR', fontsize=30)
plt.xlabel('Day since rainfall start', fontsize=30)

plt.tick_params(axis='y', labelsize=28)

# Horizontal grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# y-axis limits with a small buffer
min_val = df_ordered[dates].min().min() - 0.05
max_val = df_ordered[dates].max().max() + 0.05
plt.ylim(min_val, max_val)

# Layout adjustment
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ------------------------------------------------------------
# 6. Output
# ------------------------------------------------------------
output_path = r'YOUR_LOCAL_PATH\Grouped_bar_plot_daily_signal_change_ratios.png'
plt.savefig(output_path, dpi=400)
print(f"Figure saved to: {output_path}")

