# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe

# -----------------------------
# 1. 全局字体与大小设置
# -----------------------------
matplotlib.rcParams['font.family'] = 'Times New Roman'  # 使用 Times New Roman
matplotlib.rcParams['font.size'] = 32           # 基础字体大小
matplotlib.rcParams['axes.labelsize'] = 32      # 坐标轴 label 字体
matplotlib.rcParams['axes.titlesize'] = 32      # 图标题字体
matplotlib.rcParams['xtick.labelsize'] = 32     # x 轴刻度文字
matplotlib.rcParams['ytick.labelsize'] = 32     # y 轴刻度文字
matplotlib.rcParams['legend.fontsize'] = 32     # 图例文字

# -----------------------------
# 2. 读取 Excel 文件
# -----------------------------
file_path = r'D:\eva_zz_new\全天OD逐小时\原始信令信息.xlsx'  # 原始字符串写法，避免反斜杠转义
df = pd.read_excel(file_path)

# -----------------------------
# 3. 计算到中心网格的 2D 距离
# -----------------------------
center_coords = {
    0: (62, 118),  # Center Zhengzhou City
    1: (66, 82),   # Xingyang City & Shangjie District
    2: (58, 158),  # Zhongmou County
    3: (62, 33),   # Gongyi City
    4: (35, 81),   # Xinmi City
    5: (17, 127),  # Xinzheng City
    6: (25, 39)    # Dengfeng City
}

df['center_row'] = df['administrative district ID'].map(lambda x: center_coords[x][0])
df['center_col'] = df['administrative district ID'].map(lambda x: center_coords[x][1])

df['distance_to_center_2d'] = np.sqrt(
    (df['row'] - df['center_row']) ** 2 +
    (df['col'] - df['center_col']) ** 2
)

df['distance_to_center_2d_z'] = df.groupby('administrative district ID')['distance_to_center_2d'] \
                                  .transform(lambda x: (x - x.mean()) / x.std())

# -----------------------------
# 4. 颜色及名称
# -----------------------------
colors = [
    '#6D9B98',  # ID=0: Center Zhengzhou City
    '#D5A768',  # ID=1: Xingyang City & Shangjie
    '#4C516D',  # ID=2: Zhongmou County
    '#E3A857',  # ID=3: Gongyi City
    '#8CB78F',  # ID=4: Xinmi City
    '#547AA5',  # ID=5: Xinzheng City
    '#B4B486',  # ID=6: Dengfeng City
]

district_names = {
    0: "Center Zhengzhou",
    1: "Xingyang & Shangjie",
    2: "Zhongmou",
    3: "Gongyi",
    4: "Xinmi",
    5: "Xinzheng",
    6: "Dengfeng"
}

# —— 新增：把 & 之后自动换行（保留 & 本身），便于大字号排版 —— #
district_names_wrapped = {
    k: re.sub(r'\s*&\s*', ' &\n', v) for k, v in district_names.items()
}

# -----------------------------
# 5. 创建 pivot 表并绘图
# -----------------------------
grid_data = df.pivot(index='row', columns='col', values='administrative district ID')

plt.figure(figsize=(20, 10))
sns.heatmap(grid_data,
            cmap=sns.color_palette(colors),
            cbar=False,
            square=True)

# -----------------------------
# 6. 在图中标注行政区中心（空心 X）和名称
# -----------------------------
# 大字号+白描边，提升可读性
LABEL_FS = 26
PATH_EFFECTS = [pe.withStroke(linewidth=3, foreground="white")]

for district_id, (c_row, c_col) in center_coords.items():
    # 空心 X 散点
    plt.scatter(
        c_col, c_row,
        marker='X',
        s=120,
        facecolors='none',   # 让 X 空心
        edgecolors='black',  # X 的边缘颜色
        linewidths=1.5,
        label='_nolegend_'   # 避免自动进入图例
    )
    # 在点旁边标注行政区名称（已自动换行）
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
# 7. 给空心 X 加入图例（仅一项：District administrative center）
# -----------------------------
center_handle = plt.scatter(
    [], [],  # 占位
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
# 8. 坐标轴及标题等设置
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

# 给坐标轴加外框线
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

plt.tight_layout()

# -----------------------------
# 9. 保存图片 & 数据
# -----------------------------
output_fig_path = r'D:\eva_zz_new\任务图1月\特征工程\districts_2d_centers_2511\.png'
plt.savefig(output_fig_path, dpi=400, bbox_inches='tight')
plt.close()

output_data_path = r'D:\eva_zz_new\任务图1月\特征工程\updated_dataset_with_2d_distances_2510.xlsx'
df.to_excel(output_data_path, index=False)
