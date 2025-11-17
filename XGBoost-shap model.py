import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib
# ------------------------------------------------------------
# Explicitly use the 'Agg' backend to avoid 'tostring_rgb' errors
# in headless or remote environments.
# ------------------------------------------------------------
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    make_scorer
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import shap
from xgboost import XGBClassifier
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ============================================================
# 0. Global style and font settings (Times New Roman)
# ============================================================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False  # Ensure minus sign is displayed correctly

# ============================================================
# 1. Load dataset
# ============================================================
file_path = r'YOUR_LOCAL_PATH\dataset_randomly generated beta version.xlsx'
data = pd.read_excel(file_path)

# Predefined ordered resilience-category labels
custom_order = ['R', 'L_F', 'L_M', 'L_S', 'M_F', 'M_M', 'M_S', 'H_F', 'H_M', 'H_S']
data['target_codes'] = pd.Categorical(
    data['combined category'],
    categories=custom_order,
    ordered=True
).codes

# ============================================================
# 2. Feature preparation
# ============================================================
excluded_columns = [
    'Tid',
    'combined category',
    'row',
    'col',
    'loss category',
    'recovery category'
]
features = data.drop(excluded_columns + ['target_codes'], axis=1)

# ============================================================
# 3. Initial sample weights based on predefined error costs
# ============================================================
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

initial_sample_weights = compute_sample_weight(
    class_weight=error_costs,
    y=data['target_codes']
)

# ============================================================
# 4. Model and hyperparameters
# ============================================================
model = XGBClassifier()
best_params = {
    'colsample_bytree': 0.7461787651244297,
    'learning_rate': 0.034102546602601175,
    'max_depth': 8,
    'n_estimators': 270,
    'subsample': 0.8219993315565242
}

# ============================================================
# 5. Cross-validation scorer (not directly used below, but retained)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scorer = make_scorer(
    roc_auc_score,
    needs_proba=True,
    multi_class='ovo'
)

# ============================================================
# 6. First training and prediction (baseline weights)
# ============================================================
best_model1 = xgb.XGBClassifier(**best_params)
best_model1.fit(
    features,
    data['target_codes'],
    sample_weight=initial_sample_weights
)
predictions_1 = best_model1.predict(features)
prob_predictions_1 = best_model1.predict_proba(features)

conf_matrix_1 = confusion_matrix(data['target_codes'], predictions_1)
error_rates_1 = 1 - np.diag(conf_matrix_1) / np.sum(conf_matrix_1, axis=1)

accuracy_1 = accuracy_score(data['target_codes'], predictions_1)
roc_auc_1 = roc_auc_score(data['target_codes'], prob_predictions_1, multi_class='ovo')
f1_micro_1 = f1_score(data['target_codes'], predictions_1, average='micro')
f1_macro_1 = f1_score(data['target_codes'], predictions_1, average='macro')
mcc_1 = matthews_corrcoef(data['target_codes'], predictions_1)

print("First prediction metrics:")
print(f"Confusion matrix:\n{conf_matrix_1}")
print(f"Error rates: {error_rates_1}")
print(f"Accuracy: {accuracy_1}")
print(f"ROC AUC: {roc_auc_1}")
print(f"F1 (micro): {f1_micro_1}")
print(f"F1 (macro): {f1_macro_1}")
print(f"MCC: {mcc_1}")

# ============================================================
# 7. Neighbourhood-based weight adjustment
# ============================================================
def get_neighborhood(df: pd.DataFrame, r: int, c: int) -> pd.DataFrame:
    """
    Return the 3×3 Moore neighbourhood for the grid cell centred at (r, c),
    including the focal cell. Neighbours are defined by row and column indices
    within [r-1, r+1] and [c-1, c+1], respectively.
    """
    neighbors = df[
        (df['row'] >= r - 1) & (df['row'] <= r + 1) &
        (df['col'] >= c - 1) & (df['col'] <= c + 1)
    ]
    return neighbors


boundary_threshold = 10
mobile_signaling = data['mobile signaling in referential day']
percentile_20 = np.percentile(mobile_signaling, 20)
percentile_90 = np.percentile(mobile_signaling, 90)

adjusted_weights = []
for index, row in data.iterrows():
    neighbors = get_neighborhood(data, row['row'], row['col'])
    same_class_neighbors = neighbors[neighbors['target_codes'] == row['target_codes']]
    neighbor_count = len(same_class_neighbors)

    weight = initial_sample_weights[index]

    # Spatial-context-based reweighting
    if neighbor_count == 2:
        weight *= 1.1
    elif neighbor_count == 1:
        weight *= 1.5
    elif neighbor_count == 0:
        # Isolated cell for this resilience category
        weight *= 2.0
        # Additional emphasis for boundary cells and low signalling
        if (
            (row['col'] < boundary_threshold) or
            (row['col'] > (data['col'].max() - boundary_threshold)) or
            (row['row'] < boundary_threshold) or
            (row['row'] > (data['row'].max() - boundary_threshold)) or
            (row['mobile signaling in referential day'] <= percentile_20)
        ):
            weight *= 3.0

    # Additional emphasis for the upper 10% of mobile signalling values
    if row['mobile signaling in referential day'] >= percentile_90:
        weight *= 1.2

    adjusted_weights.append(weight)

if len(adjusted_weights) == len(data):
    data['sample_weights'] = adjusted_weights
else:
    raise ValueError(
        "The length of adjusted_weights does not match the length of the dataset."
    )

# ============================================================
# 8. Second training and prediction (spatially adjusted weights)
# ============================================================
best_model2 = xgb.XGBClassifier(**best_params)
best_model2.fit(
    features,
    data['target_codes'],
    sample_weight=data['sample_weights']
)
predictions_2 = best_model2.predict(features)
prob_predictions_2 = best_model2.predict_proba(features)

conf_matrix_2 = confusion_matrix(data['target_codes'], predictions_2)
error_rates_2 = 1 - np.diag(conf_matrix_2) / np.sum(conf_matrix_2, axis=1)

accuracy_2 = accuracy_score(data['target_codes'], predictions_2)
roc_auc_2 = roc_auc_score(data['target_codes'], prob_predictions_2, multi_class='ovo')
f1_micro_2 = f1_score(data['target_codes'], predictions_2, average='micro')
f1_macro_2 = f1_score(data['target_codes'], predictions_2, average='macro')
mcc_2 = matthews_corrcoef(data['target_codes'], predictions_2)

print("Second prediction metrics:")
print(f"Confusion matrix:\n{conf_matrix_2}")
print(f"Error rates: {error_rates_2}")
print(f"Accuracy: {accuracy_2}")
print(f"ROC AUC: {roc_auc_2}")
print(f"F1 (micro): {f1_micro_2}")
print(f"F1 (macro): {f1_macro_2}")
print(f"MCC: {mcc_2}")

# ============================================================
# 9. Select final model based on higher training-set accuracy
# ============================================================
if accuracy_2 > accuracy_1:
    best_model = best_model2
    predictions = predictions_2
    prob_predictions = prob_predictions_2
    conf_matrix = conf_matrix_2
    error_rates = error_rates_2
    accuracy = accuracy_2
    roc_auc = roc_auc_2
    f1_micro = f1_micro_2
    f1_macro = f1_macro_2
    mcc = mcc_2
else:
    best_model = best_model1
    predictions = predictions_1
    prob_predictions = prob_predictions_1
    conf_matrix = conf_matrix_1
    error_rates = error_rates_1
    accuracy = accuracy_1
    roc_auc = roc_auc_1
    f1_micro = f1_micro_1
    f1_macro = f1_macro_1
    mcc = mcc_1

print("Selected model metrics (based on higher training accuracy):")
print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")
print(f"F1 (micro): {f1_micro}")
print(f"F1 (macro): {f1_macro}")
print(f"MCC: {mcc}")

# ============================================================
# 10. SHAP value computation
# ============================================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(features)
print(f"shap_values shape: {shap_values.shape}")

# Compute the mean absolute SHAP values and sum across classes
shap_sum = np.abs(shap_values).mean(axis=0)
shap_sums = shap_sum.sum(axis=1)
sorted_indices = np.argsort(shap_sums)[::-1]

sorted_feature_names = np.array(features.columns)[sorted_indices]
sorted_shap_sum = shap_sum[sorted_indices]
print(f"sorted_shap_sum shape: {sorted_shap_sum.shape}")

# ============================================================
# 11. Reorder SHAP values by the predefined resilience-category order
# ============================================================
class_index_map = {name: index for index, name in enumerate(custom_order)}
reordered_shap_values = np.zeros((sorted_shap_sum.shape[0], len(custom_order)))

for i in range(sorted_shap_sum.shape[0]):
    for j in range(sorted_shap_sum.shape[1]):
        class_name = custom_order[j]
        reordered_shap_values[i, class_index_map[class_name]] = sorted_shap_sum[i, j]

# Percentage contribution of each class to the SHAP value of each feature
percentages = np.round(
    np.array(reordered_shap_values) /
    np.sum(reordered_shap_values, axis=1, keepdims=True) * 100,
    4
)

percentages_df = pd.DataFrame(
    percentages,
    columns=custom_order,
    index=sorted_feature_names
)

csv_save_path = r'YOUR_LOCAL_PATH\shap_values_percentages_resilience250206.csv'
percentages_df.to_csv(csv_save_path)
print(f"SHAP percentage table saved to: {csv_save_path}")

# ============================================================
# 12. Stacked bar plotting of class-wise SHAP values per feature
# ============================================================
colors = [
    '#4C516D',  # R
    '#547AA5',  # L_F
    '#5E7D7E',  # L_M
    '#6D9B98',  # L_S
    '#8CB78F',  # M_F
    '#B4B486',  # M_M
    '#D5A768',  # M_S
    '#E3A857',  # H_F
    '#DC8744',  # H_M
    '#C95D63'   # H_S
]

# Manual mapping between feature letters and original variable names
manual_pairs = [
    ("A", "mobile signaling in referential day"),
    ("B", "slope"),
    ("C", "light intensity"),
    ("D", "max daily rainfall"),
    ("E", "standardized distance to distract 3d center"),
    ("F", "total rainfall"),
    ("G", "annual GDP(million RMB)"),
    ("H", "distance to the closet highway/main road(km)"),
    ("I", "standardized distance to distract 2d center"),
    ("J", "population density(people/km2)"),
    ("K", "permeability"),
    ("L", "building cover rate"),
    ("M", "transportation facility"),
    ("N", "urban ring zone")
]

letters = [pair[0] for pair in manual_pairs]
feature_mapping = {pair[0]: pair[1] for pair in manual_pairs}
feature_to_letter = {v: k for k, v in feature_mapping.items()}

# Replace feature names with corresponding letters where applicable
sorted_letters = [feature_to_letter.get(name, name) for name in sorted_feature_names]

fig, ax = plt.subplots(figsize=(15, 12))

# Cumulative SHAP values for annotation
cumulative_shap = np.cumsum(reordered_shap_values, axis=1)

# Stacked bar plot: each bar corresponds to one feature, stacked by class
for i, color in zip(range(len(custom_order)), colors):
    ax.bar(
        sorted_letters,
        reordered_shap_values[:, i],
        bottom=np.sum(reordered_shap_values[:, :i], axis=1),
        color=color,
        edgecolor='white',
        label=custom_order[i]
    )

# Numeric labels for total SHAP values (sum across classes) on each bar
for i, letter in enumerate(sorted_letters):
    total_value = cumulative_shap[i, -1]
    ylim_top = ax.get_ylim()[1] - 0.2

    if total_value < 1.5:
        offset = 0.05
    elif 1.5 <= total_value <= 3.0:
        offset = 0.05
    else:
        offset = 0.0

    y_pos = total_value if total_value < ylim_top else ylim_top
    ax.text(
        i,
        y_pos + offset,
        f'{total_value:.3f}',
        va='bottom',
        ha='center',
        fontsize=24,
        color='black'
    )

# x-axis configuration
ax.set_xticks(range(len(sorted_letters)))
ax.set_xticklabels(
    sorted_letters,
    rotation=0,
    ha='right',
    fontsize=24,
    fontweight='bold'
)

# Axis labels and title
ax.set_ylabel('Average SHAP value', fontsize=24, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
ax.tick_params(axis='y', labelsize=24)
ax.set_xlabel('Features', fontsize=24, fontweight='bold')
ax.set_title(
    'SHAP Values per Class for Each Feature in Ten-Class Resilience Prediction',
    fontsize=28,
    fontweight='bold'
)

# Internal legend (for resilience categories R–H_S), reversed so H_S is at the top
internal_handles = [
    Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(custom_order))
]
internal_labels = custom_order.copy()
internal_handles = internal_handles[::-1]
internal_labels = internal_labels[::-1]

internal_legend = ax.legend(
    handles=internal_handles,
    labels=internal_labels,
    title='Resilience classes',
    loc='upper right',
    fontsize=28,
    title_fontsize=36,
    frameon=True
)
ax.add_artist(internal_legend)

# External legend (feature letter–name mapping)
external_legend_elements = [
    Line2D(
        [0],
        [0],
        marker='s',
        color='w',
        label=f"{letter}: {name}",
        markerfacecolor='none',
        markersize=10
    )
    for letter, name in manual_pairs
]

external_legend = fig.legend(
    handles=external_legend_elements,
    title='Legend: feature–letter mapping',
    bbox_to_anchor=(1.02, 0.5),
    loc='center left',
    fontsize=32,
    title_fontsize=36,
    frameon=True
)

# Adjust layout to accommodate the external legend on the right
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save figure
save_path = r'YOUR_LOCAL_PATH\SHAP_stacked_bar_ten_class_resilience.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved to: {save_path}")
