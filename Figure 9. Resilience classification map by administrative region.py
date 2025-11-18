import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# ------------------------------------------------------------
# Backend setting
# ------------------------------------------------------------
# Use the non-interactive 'Agg' backend to avoid rendering errors
# (e.g. 'tostring_rgb' errors) in certain headless or remote environments.
matplotlib.use('Agg')

from decimal import Decimal, getcontext

# ------------------------------------------------------------
# Global font configuration
# ------------------------------------------------------------
# Use Times New Roman as the default font for all text elements.
plt.rcParams['font.family'] = 'Times New Roman'

# Set high-precision decimal arithmetic for robust normalisation.
getcontext().prec = 10

# ------------------------------------------------------------
# Input Excel file path
# ------------------------------------------------------------
file_path = r'YOUR_LOCAL_PATH\dataset_randomly generated beta version.xlsx'

# Read the Excel file containing resilience-category compositions.
try:
    df_excel = pd.read_excel(file_path, sheet_name='Sheet1')
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find file: {file_path}. Please check the path.")
except Exception as e:
    raise Exception(f"Error reading Excel file: {e}")

print("Columns in Excel:")
print(df_excel.columns.tolist())

if 'combined category' not in df_excel.columns:
    raise KeyError("Column 'combined category' not found in Excel file. Please check the Excel header.")

# Use 'combined category' as index for subsequent processing.
df_excel.set_index('combined category', inplace=True)

# Transpose so that each administrative unit (code) becomes a row.
df_transposed = df_excel.transpose()
print("Transposed DataFrame:")
print(df_transposed.head())

# ------------------------------------------------------------
# Mapping from code to full administrative unit names
# ------------------------------------------------------------
code_to_city = {
    0: 'Central Zhengzhou City',
    1: 'Xingyang City & Shangjie District',
    2: 'Zhongmou City',
    3: 'Gongyi City',
    4: 'Xinmi City',
    5: 'Xinzheng City',
    6: 'Dengfeng City',
    'total': 'Zhengzhou City'
}

# Reorganise the transposed DataFrame into a city-based structure.
data = {}
for code, row in df_transposed.iterrows():
    if code in code_to_city:
        city = code_to_city[code]
        data[city] = row.to_dict()
    else:
        print(f"Unrecognised code: {code}")

df = pd.DataFrame(data).T
print("Processed DataFrame (raw category counts or proportions):")
print(df.head())

# ------------------------------------------------------------
# 1) Resilience category labels (10 levels, ordered from lowest to highest)
# ------------------------------------------------------------
categories = [
    'R',      # lowest resilience
    'L_F',
    'L_M',
    'L_S',
    'M_F',
    'M_M',
    'M_S',
    'H_F',
    'H_M',
    'H_S'     # highest resilience
]

# ------------------------------------------------------------
# 2) Colour palette for each resilience category
# ------------------------------------------------------------
colors = [
    '#4C516D',  # R
    '#547AA5',  # L_F
    '#5E7D7E',  # L_M
    '#6D9B98',  # L_S
    '#8CB78F',  # M_F
    '#B4B486',  # M_M
    '#D5A768',  # M_S
    '#E3A857',  # H_F (alternative)
    '#DC8744',  # H_M
    '#C95D63'   # H_S
]

# Check that all resilience-category columns are present in the DataFrame.
missing_categories = [cat for cat in categories if cat not in df.columns]
if missing_categories:
    raise KeyError(
        f"Missing category columns in Excel: {missing_categories}. "
        f"Please check the column names."
    )
else:
    print("All resilience-category columns are present.")

# Replace any missing values (NaNs) by zeros prior to normalisation.
df.fillna(0, inplace=True)

# ------------------------------------------------------------
# Row-wise normalisation to ensure each administrative unit sums to 1
# ------------------------------------------------------------
df_normalized = df.div(df.sum(axis=1), axis=0)
df_normalized = df_normalized.applymap(lambda x: Decimal(str(x)))

# Ensure that the last category ('H_S') is adjusted such that the
# row sum is exactly 1.0 after high-precision arithmetic.
last_category = 'H_S'
for index, row in df_normalized.iterrows():
    cumulative_sum = sum(row[cat] for cat in categories[:-1])
    row[last_category] = Decimal('1') - cumulative_sum
    df_normalized.loc[index] = row

# Convert back to float for plotting.
df_normalized = df_normalized.applymap(float)
print("Normalised DataFrame (row sums should be 1.0):")
print(df_normalized.head())

sums = df_normalized.sum(axis=1)
print("Row sums (should be 1.0):")
print(sums)

if not all(np.isclose(sums, 1.0, atol=1e-8)):
    print("Warning: some rows do not sum to 1.0 exactly after normalisation.")
else:
    print("All rows sum to 1.0 within numerical tolerance.")

# ------------------------------------------------------------
# Desired plotting order along the x-axis (internal indexing by full names)
# ------------------------------------------------------------
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

missing_cities = [city for city in desired_order if city not in df_normalized.index]
if missing_cities:
    print(f"Warning: the following administrative units are missing and will be excluded: {missing_cities}")

df_normalized = df_normalized.reindex(desired_order)
print("Reordered normalised DataFrame:")
print(df_normalized)

print("Check row sums after reordering:")
print(df_normalized.sum(axis=1))

# ------------------------------------------------------------
# Figure and axis configuration
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 11))

# Array storing the cumulative height of stacked segments.
bottom_val = np.zeros(len(df_normalized))

# Construct stacked bar chart: each bar represents one administrative unit.
for cat, color in zip(categories, colors):
    values = df_normalized[cat].values
    bars = ax.bar(
        df_normalized.index,
        values,
        bottom=bottom_val,
        label=cat,
        color=color
    )

    # Add percentage labels (one decimal place) at the centre of each segment.
    ax.bar_label(
        bars,
        labels=[f"{v * 100:.1f}%" if not np.isnan(v) else '' for v in values],
        label_type='center',
        fontsize=18,
        color='black'
    )

    bottom_val += values

print("Final 'bottom_val' (should be 1.0 for all administrative units):")
print(bottom_val)

# ------------------------------------------------------------
# Axis labels, tick configuration, and figure title
# ------------------------------------------------------------
ax.set_ylabel("Proportion of grids", fontsize=26)

# Academic English title corresponding to “分行政区韧性类别组成图”
ax.set_title(
    "Composition of Resilience Categories by Administrative District",
    fontsize=30,
    pad=18
)

# Short, reader-friendly labels for the x-axis (only labels, not used for indexing).
short_xticks = [
    'Zhengzhou',
    'Xinmi',
    'Dengfeng',
    'Zhongmou',
    'Xinzheng',
    'Central Zhengzhou',
    'Xingyang & Shangjie',
    'Gongyi'
]

# Ensure that the number of labels matches the number of x positions.
ax.set_xticks(range(len(short_xticks)))
ax.set_xticklabels(short_xticks, rotation=45, ha='right', fontsize=24)

# Configure tick label sizes for both axes.
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)

# y-axis ticks and limits.
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0, 1.0)

# Optional horizontal grid lines for improved readability.
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Legend for resilience categories.
legend = ax.legend(
    title="Resilience categories",
    title_fontsize=24,
    bbox_to_anchor=(1.05, 1.0),
    loc='upper left',
    fontsize=20
)

plt.tight_layout()

# ------------------------------------------------------------
# Output: save as TIFF and PNG (300 dpi)
# ------------------------------------------------------------
output_path_tif = r'YOUR_LOCAL_PATH\Resilience_category_composition_by_district.tif'
plt.savefig(
    output_path_tif,
    dpi=300,
    format='tif',
    bbox_inches='tight',
    pil_kwargs={'compression': 'none'}
)

output_path_png = r'YOUR_LOCAL_PATH\Resilience_category_composition_by_district.png'
plt.savefig(
    output_path_png,
    dpi=300,
    format='png',
    bbox_inches='tight'
)

plt.show()
print(f"Saved: {output_path_tif}")
print(f"Saved: {output_path_png}")
