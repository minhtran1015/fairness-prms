import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('evaluation_output/evaluation_results.csv')

# --- Từ điển dịch tiếng Việt ---
translation_map = {
    "Disability_status": "Tình trạng khuyết tật",
    "SES": "Tình trạng KT-XH",
    "Race_x_gender": "Chủng tộc & Giới tính",
    "Age": "Tuổi tác",
    "Race_ethnicity": "Chủng tộc & Sắc tộc",
    "Race_x_SES": "Chủng tộc & KT-XH",
    "Gender_identity": "Bản dạng giới",
    "Religion": "Tôn giáo",
    "Nationality": "Quốc tịch",
    "Sexual_orientation": "Xu hướng tính dục",
    "Physical_appearance": "Ngoại hình"
}

temp_map = {
    'temp_001': '0.01',
    'temp_02': '0.2',
    'temp_04': '0.4',
    'temp_08': '0.8'
}

# Áp dụng bản dịch
df['category_vn'] = df['category'].map(translation_map)
df['temp_label'] = df['temp_setting'].map(temp_map)

# ===== LOẠI BỎ SES và Disability_status vì chúng có metrics = 0 (dữ liệu không hợp lệ) =====
df_cleaned = df[~df['category'].isin(['SES', 'Disability_status'])].copy()
print(f"⚠️  Đã loại bỏ SES và Disability_status (metrics không hợp lệ)")
print(f"📊 Categories được giữ lại: {sorted(df_cleaned['category'].unique().tolist())}")

# Tạo pivot table cho heatmap
pivot_data = df_cleaned.pivot(index='category_vn', columns='temp_label', values='eodds_gap')

# Sắp xếp categories theo giá trị trung bình EOdds Gap
avg_eodds = pivot_data.mean(axis=1).sort_values(ascending=False)
pivot_data = pivot_data.loc[avg_eodds.index]

# Sắp xếp cột temperatures theo thứ tự tăng dần
pivot_data = pivot_data[['0.01', '0.2', '0.4', '0.8']]

# Tạo figure với 2 subplots sử dụng GridSpec để tăng khoảng cách
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35)

# ============ SUBPLOT 1: HEATMAP ============
ax1 = fig.add_subplot(gs[0, 0])

# Vẽ heatmap với màu sắc gradient
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
            cbar_kws={'label': 'EOdds Gap'}, linewidths=0.5,
            ax=ax1, vmin=0, vmax=pivot_data.max().max(),
            annot_kws={'size': 11})

ax1.set_xlabel('Temperature', fontsize=16)
ax1.set_ylabel('Loại Thiên Kiến', fontsize=16)
ax1.tick_params(axis='both', labelsize=12)

# Xoay labels
plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')
plt.setp(ax1.get_yticklabels(), rotation=0)

# ============ SUBPLOT 2: GROUPED BAR CHART ============
ax2 = fig.add_subplot(gs[0, 1])

# Chuẩn bị dữ liệu cho grouped bar chart
x = np.arange(len(pivot_data.index))
width = 0.2

colors = ['#fee0d2', '#fcbba1', '#fc9272', '#de2d26']
temps = ['0.01', '0.2', '0.4', '0.8']

for i, temp in enumerate(temps):
    offset = width * (i - 1.5)
    bars = ax2.barh(x + offset, pivot_data[temp], width, 
                     label=f'Temp {temp}', color=colors[i], alpha=0.8)

ax2.set_ylabel('Loại Thiên Kiến', fontsize=16)
ax2.set_xlabel('EOdds Gap (↓)', fontsize=16)
ax2.set_yticks(x)
ax2.set_yticklabels(pivot_data.index, fontsize=11)
ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax2.grid(axis='x', linestyle='--', alpha=0.3)
ax2.tick_params(axis='x', labelsize=11)

plt.savefig('evaluation_output/eodds_gap_all_temps_cleaned.png', 
            dpi=300, bbox_inches='tight')

print("\n✅ Biểu đồ EOdds Gap (đã loại bỏ 2 category) đã được lưu!")
print(f"📊 Số loại thiên kiến: {len(pivot_data.index)}")
print(f"🌡️  Số mức temperature: {len(temps)}")
print(f"\n📈 Giá trị EOdds Gap cao nhất: {pivot_data.max().max():.3f}")
print(f"📉 Giá trị EOdds Gap thấp nhất: {pivot_data.min().min():.3f}")
