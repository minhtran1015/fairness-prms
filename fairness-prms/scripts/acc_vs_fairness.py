import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

# --- Cài đặt cỡ chữ (Tập trung ở đây) ---
AXIS_LABEL_FONT = 21  # Cỡ chữ cho tiêu đề trục X, Y (Giữ nguyên)
TICK_LABEL_FONT = 16  # Cỡ chữ cho các số trên trục (VD: 0.01, 0.2, 0.55)
DATA_LABEL_FONT = 18  # Cỡ chữ cho các số trên đường line (VD: 0.568)
LEGEND_FONT = 18      # Cỡ chữ cho chú thích (Legend)
# --- Kết thúc cài đặt ---

df = pd.read_csv('fairness-prms/evaluation_output/evaluation_results.csv')

# Sửa lại thứ tự sắp xếp của temperature
df['temp_numeric'] = df['temp_setting'].str.replace('temp_', '').str.replace('001', '0.01').str.replace('02', '0.2').str.replace('04', '0.4').str.replace('08', '0.8').astype(float)
df = df.sort_values('temp_numeric')
temp_order = df['temp_setting'].unique() 

# Nhóm theo 'temp_setting' và tính trung bình của 3 chỉ số
df_agg = df.groupby('temp_setting')[['accuracy', 'eopp_gap', 'eodds_gap']].mean().reset_index()

# Sắp xếp lại DataFrame tổng hợp theo đúng thứ tự nhiệt độ
df_agg = df_agg.set_index('temp_setting').loc[temp_order].reset_index()
df_agg['temp_labels'] = ['0.01', '0.2', '0.4', '0.8']

# --- Bắt đầu vẽ biểu đồ ---
plt.style.use('seaborn-v0_8-talk') 
fig, ax1 = plt.subplots(figsize=(12, 10)) 

ax1.set_xlabel('Nhiệt Độ Sinh ($\\mathcal{T}$)', fontsize=AXIS_LABEL_FONT, labelpad=10)
ax1.tick_params(axis='x', labelsize=TICK_LABEL_FONT) # <--- THAY ĐỔI: Thêm cỡ chữ cho trục X

# --- Trục Y bên trái (Accuracy) ---
color_acc = 'tab:blue'
ax1.set_ylabel('Acc (↑)', color=color_acc, fontsize=AXIS_LABEL_FONT)
ax1.plot(df_agg['temp_labels'], df_agg['accuracy'], color=color_acc, marker='o', linestyle='-', linewidth=3, markersize=10, label='Acc (Trục trái)')
ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=TICK_LABEL_FONT) # <--- THAY ĐỔI: Thêm cỡ chữ
ax1.set_ylim(0.55, 0.70) 

# Thêm nhãn dữ liệu cho Accuracy
for i, txt in enumerate(df_agg['accuracy']):
    ax1.annotate(f'{txt:.3f}', (df_agg['temp_labels'][i], df_agg['accuracy'][i]), 
                 textcoords="offset points", 
                 xytext=(0,15), 
                 ha='center', 
                 color=color_acc,
                 fontsize=DATA_LABEL_FONT) # <--- THAY ĐỔI: Thêm cỡ chữ

# --- Trục Y bên phải (Gaps) ---
ax2 = ax1.twinx() 
color_eopp = 'tab:orange'
color_eodds = 'tab:green'

ax2.set_ylabel('Chênh Lệch Trung Bình (↓)', color='gray', fontsize=AXIS_LABEL_FONT)
# Vẽ đường EOpp Gap
ax2.plot(df_agg['temp_labels'], df_agg['eopp_gap'], color=color_eopp, marker='s', linestyle='--', linewidth=2, markersize=8, label='EOpp Gap (Trục phải)')
# Vẽ đường EOdds Gap
ax2.plot(df_agg['temp_labels'], df_agg['eodds_gap'], color=color_eodds, marker='^', linestyle=':', linewidth=2, markersize=8, label='EOdds Gap (Trục phải)')
ax2.tick_params(axis='y', labelcolor='gray', labelsize=TICK_LABEL_FONT) # <--- THAY ĐỔI: Thêm cỡ chữ
ax2.set_ylim(0, 0.15) 

# Thêm nhãn dữ liệu cho Gaps
for i, txt in enumerate(df_agg['eopp_gap']):
    ax2.annotate(f'{txt:.3f}', (df_agg['temp_labels'][i], df_agg['eopp_gap'][i]), 
                 textcoords="offset points", 
                 xytext=(0,-20), 
                 ha='center', 
                 color=color_eopp,
                 fontsize=DATA_LABEL_FONT) # <--- THAY ĐỔI: Thêm cỡ chữ
for i, txt in enumerate(df_agg['eodds_gap']):
    ax2.annotate(f'{txt:.3f}', (df_agg['temp_labels'][i], df_agg['eodds_gap'][i]), 
                 textcoords="offset points", 
                 xytext=(0,15), 
                 ha='center', 
                 color=color_eodds,
                 fontsize=DATA_LABEL_FONT) # <--- THAY ĐỔI: Thêm cỡ chữ

# --- Chú thích (Legend) ---
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines + lines2, labels + labels2, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), 
           ncol=3, 
           fontsize=LEGEND_FONT) # <--- THAY ĐỔI: Dùng biến cỡ chữ riêng

# Tinh chỉnh layout và lưu file
fig.tight_layout(rect=(0, 0, 1, 0.93)) 
plt.savefig('fairness-prms/evaluation_output/combined_accuracy_vs_gaps.svg')
plt.savefig('fairness-prms/evaluation_output/combined_accuracy_vs_gaps.png')
plt.savefig('fairness-prms/evaluation_output/combined_accuracy_vs_gaps.pdf')

print("\nBiểu đồ kết hợp với phông chữ LỚN HƠN đã được lưu vào file 'combined_accuracy_vs_gaps.svg'")