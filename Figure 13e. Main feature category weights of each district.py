import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ------------------------------------------------------------
# 0. Global settings
# ------------------------------------------------------------
mpl.rcParams['font.family'] = 'Times New Roman'

# Factor-category order (from bottom to top in the stacked bar chart)
categories = [
    'Spatial Information',
    'Geographical Information',
    'Rainfall Information',
    'Infrastructure and Transport',
    'Economic and Population'
]

# Colour mapping for each main factor category
main_category_colors = {
    'Spatial Information': '#5E7D7E',
    'Geographical Information': '#6D9B98',
    'Rainfall Information': '#4C516D',
    'Infrastructure and Transport': '#8CB78F',
    'Economic and Population': '#E3A857'
}

# ------------------------------------------------------------
# 1. Load and pre-process data
# ------------------------------------------------------------
file_path = r'YOUR_LOCAL_PATH\dataset_randomly generated beta version.xlsx'
data = pd.read_excel(file_path)
data.rename(columns={'administrative district ID': 'district_id'}, inplace=True)

# Mapping of main factor categories to individual variables
main_cat_dict = {
    'Spatial Information': [
        'urban ring zone',
        'standardized distance to distract 3d center',
        'standardized distance to distract 2d center'
    ],
    'Rainfall Information': [
        'max daily rainfall',
        'total rainfall'
    ],
    'Geographical Information': [
        'slope',
        'permeability'
    ],
    'Infrastructure and Transport': [
        'light intensity',
        'transportation facility',
        'distance to the closet highway/main road(km)',
        'building cover rate'
    ],
    'Economic and Population': [
        'mobile signaling in referential day',
        'population density(people/km2)',
        'annual GDP(million RMB)'
    ]
}

# Ordered multi-class resilience categories
custom_order = ['R', 'L_F', 'L_M', 'L_S', 'M_F', 'M_M', 'M_S', 'H_F', 'H_M', 'H_S']
data['target_codes'] = pd.Categorical(
    data['combined category'],
    categories=custom_order,
    ordered=True
).codes

# Excluded variables (IDs, coordinates, labels)
excluded_columns = [
    'Tid',
    'combined category',
    'row',
    'col',
    'loss category',
    'recovery category',
    'district_id'
]

# Feature matrix (all explanatory variables)
features = data.drop(excluded_columns + ['target_codes'], axis=1)

# ------------------------------------------------------------
# 2. Sample-weight specification
# ------------------------------------------------------------
# Misclassification costs for each resilience category (by code)
error_costs = {0: 1.3, 1: 2, 2: 1.6, 3: 1.5, 4: 1.5, 5: 1.2, 6: 1.6, 7: 1, 8: 1.1, 9: 1.3}

# Initial sample weights according to the predefined misclassification costs
initial_sample_weights = compute_sample_weight(
    class_weight=error_costs,
    y=data['target_codes']
)

def get_neighborhood(df, row_, col_):
    """
    Identify the 3×3 Moore neighbourhood centred on (row_, col_),
    including the focal grid cell itself.
    """
    return df[
        (df['row'] >= row_ - 1) & (df['row'] <= row_ + 1) &
        (df['col'] >= col_ - 1) & (df['col'] <= col_ + 1)
    ]

boundary_threshold = 10
mobile_signaling_col = 'mobile signaling in referential day'
p20 = np.percentile(data[mobile_signaling_col], 20)
p90 = np.percentile(data[mobile_signaling_col], 90)

# Replace missing values with zeros before weight refinement
data.fillna(0, inplace=True)

adjusted_weights = []
for i, r in data.iterrows():
    neighbors = get_neighborhood(data, r['row'], r['col'])
    same_class_neighbors = neighbors[neighbors['target_codes'] == r['target_codes']]
    neighbor_count = len(same_class_neighbors)

    w = initial_sample_weights[i]

    # Spatial-context-based weight adjustment
    if neighbor_count == 2:
        w *= 1.1
    elif neighbor_count == 1:
        w *= 1.5
    elif neighbor_count == 0:
        # Isolated cell of its resilience category
        w *= 2
        # Additional penalty for boundary/low-signalling conditions
        if (
            r['col'] < boundary_threshold or
            r['col'] > (data['col'].max() - boundary_threshold) or
            r['row'] < boundary_threshold or
            r['row'] > (data['row'].max() - boundary_threshold) or
            r[mobile_signaling_col] <= p20
        ):
            w *= 3

    # Additional emphasis for cells with very high mobile signalling
    if r[mobile_signaling_col] >= p90:
        w *= 1.2

    adjusted_weights.append(w)

if len(adjusted_weights) != len(data):
    raise ValueError("Length of adjusted_weights is inconsistent with the number of samples in data.")

data['sample_weights'] = adjusted_weights

# ------------------------------------------------------------
# 3. District-specific XGBoost modelling
# ------------------------------------------------------------
district_ids = sorted(data['district_id'].unique())

# Pre-optimised hyperparameters for XGBoost
best_params = {
    'colsample_bytree': 0.7461787651244297,
    'learning_rate': 0.034102546602601175,
    'max_depth': 8,
    'n_estimators': 270,
    'subsample': 0.8219993315565242
}

district_feature_importances = {}

for d_id in district_ids:
    d_df = data[data['district_id'] == d_id]

    # Skip districts with insufficient sample size
    if len(d_df) < 10:
        print(f"Warning: District ID {d_id} has too few samples; skipped.")
        continue

    X = d_df.drop(excluded_columns + ['target_codes'], axis=1)
    y = d_df['target_codes']
    sw = d_df['sample_weights']

    model = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X, y, sample_weight=sw)

    f_imp = model.feature_importances_
    f_names = X.columns
    district_feature_importances[d_id] = dict(zip(f_names, f_imp))

# ------------------------------------------------------------
# 4. Aggregate variable importance by main factor categories
#    and row-wise normalisation
# ------------------------------------------------------------
feature_importances_df = pd.DataFrame(district_feature_importances).T

# Sum importance scores within each main factor category
for cat in main_cat_dict:
    feats_in_cat = [f for f in main_cat_dict[cat] if f in feature_importances_df.columns]
    feature_importances_df[cat] = feature_importances_df[feats_in_cat].sum(axis=1, skipna=True)

# Extract the aggregated contributions for the predefined factor-category order
district_contributions = feature_importances_df[categories].copy().fillna(0)

# Row-wise normalisation to obtain proportional contributions
sum_val = district_contributions.sum(axis=1)
df_normalized = district_contributions.div(sum_val, axis=0).reindex(columns=categories)

# Ensure that the last factor category is adjusted to make the row sum exactly 1
last_cat = categories[-1]
for idx, row_ in df_normalized.iterrows():
    cumulative_sum = row_[categories[:-1]].sum()
    df_normalized.loc[idx, last_cat] = 1.0 - cumulative_sum

# ------------------------------------------------------------
# 5. District naming (full names for ordering, simplified labels for plotting)
# ------------------------------------------------------------
desired_order = [
    'Zhongmou City',
    'Xinzheng City',
    'Center Zhengzhou City',
    'Xinmi City',
    'Dengfeng City',
    'Xingyang City & Shangjie District',
    'Gongyi City'
]

# Map district index (in df_normalized) to full administrative names
district_id_to_name = {i: nm for i, nm in enumerate(desired_order)}

def simplify_display(name: str) -> str:
    """
    Simplify the district name for display purposes:
    (1) Remove the suffix 'City' and 'District';
    (2) Convert 'Center Zhengzhou' to 'Central Zhengzhou'.
    """
    name_simple = name.replace(' City', '').replace(' District', '')
    name_simple = name_simple.replace('Center Zhengzhou', 'Central Zhengzhou')
    return name_simple

dom_cat_df = df_normalized.copy()
dom_cat_df['district_id'] = dom_cat_df.index
dom_cat_df['district_name_full'] = dom_cat_df['district_id'].map(district_id_to_name)
dom_cat_df = dom_cat_df[dom_cat_df['district_name_full'].notna()].copy()
dom_cat_df['district_display'] = dom_cat_df['district_name_full'].apply(simplify_display)

# Preserve the user-defined ordering of districts
dom_cat_df['district_name_full'] = pd.Categorical(
    dom_cat_df['district_name_full'],
    categories=desired_order,
    ordered=True
)
dom_cat_df = dom_cat_df.sort_values('district_name_full').reset_index(drop=True)

# ------------------------------------------------------------
# 6. Plotting: stacked bar chart of main factor-category contributions
#    for each administrative district
# ------------------------------------------------------------
FIG_W, FIG_H = 16, 6
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# Layout margins: provide sufficient space for legend (right) and rotated x labels (bottom)
plt.subplots_adjust(left=0.08, right=0.82, top=0.97, bottom=0.18)

colors = [main_category_colors[cat] for cat in categories]
bottom_val = np.zeros(len(dom_cat_df))

for cat, color in zip(categories, colors):
    vals = dom_cat_df[cat].values
    bars = ax.bar(
        dom_cat_df['district_display'],
        vals,
        bottom=bottom_val,
        label=cat,
        color=color
    )

    # Percentage labels (one decimal place) inside each stacked segment
    ax.bar_label(
        bars,
        labels=[f"{v * 100:.1f}%" for v in vals],
        label_type='center',
        fontsize=18,
        color='black'
    )

    bottom_val += vals

# x-axis tick labels (simplified district names)
ax.set_xticks(range(len(dom_cat_df['district_display'])))
ax.set_xticklabels(
    dom_cat_df['district_display'],
    rotation=30,
    ha='right',
    fontsize=20
)

ax.set_ylabel("Proportion", fontsize=22)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.tick_params(axis='y', labelsize=20)

# No title is used for this figure (can be added if needed)
ax.set_title("", fontsize=1, pad=0)

# Legend on the right-hand side, with reversed category order for readability
legend_categories = list(reversed(categories))
patches = [
    mpatches.Patch(color=main_category_colors[cat], label=cat)
    for cat in legend_categories
]
ax.legend(
    handles=patches,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    fontsize=20,
    title="Main category",
    title_fontsize=22,
    labelspacing=1.2,
    borderaxespad=0.8
)

plt.tight_layout()

# ------------------------------------------------------------
# 7. Save figure
# ------------------------------------------------------------
output_path = r'YOUR_LOCAL_PATH\District_level_main_category_contributions.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

