import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# --- THAY ĐỔI: Cài đặt font hỗ trợ Tiếng Việt ---
# Chỉ định font 'DejaVu Sans' (rất phổ biến) để hiển thị Unicode
# mpl.rcParams['font.family'] = 'DejaVu Sans'
# mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana']
# -----------------------------------------------

# --- Cài đặt cỡ chữ (Tập trung ở đây) ---
AXIS_LABEL_FONT = 21  # Cỡ chữ cho tiêu đề trục X, Y
TICK_LABEL_FONT = 16  # Cỡ chữ cho các hạng mục (trục Y) và số (trục X)
DATA_LABEL_FONT = 18  # Cỡ chữ cho các số ở cuối thanh (0.224, 0.200...)
# --- Kết thúc cài đặt ---

df = pd.read_csv('fairness-prms/evaluation_output/evaluation_results.csv')

# 1. Lọc dữ liệu chỉ cho temp=0.8
df_temp_08 = df[df['temp_setting'] == 'temp_08'].copy()

# --- THAY ĐỔI: Tạo từ điển dịch thuật ---
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

# Áp dụng bản dịch vào một cột mới
df_temp_08['category_vietnamese'] = df_temp_08['category'].map(translation_map)
# -----------------------------------------

# 2. Sắp xếp các hạng mục theo 'eodds_gap' tăng dần (để vẽ thanh)
df_temp_08 = df_temp_08.sort_values('eodds_gap', ascending=True)

# 3. Bắt đầu vẽ biểu đồ
plt.style.use('seaborn-v0_8-talk')
fig, ax = plt.subplots(figsize=(12, 10)) # Kích thước (width, height)

# --- THAY ĐỔI: Sử dụng cột tiếng Việt để vẽ ---
bars = ax.barh(df_temp_08['category_vietnamese'], df_temp_08['eodds_gap'], color='tab:red', alpha=0.7)
# ---------------------------------------------

# 4. Tinh chỉnh
ax.set_xlabel('EOdds Gap (↓)', fontsize=AXIS_LABEL_FONT) 
ax.set_ylabel('Loại Thiên Kiến', fontsize=AXIS_LABEL_FONT) 
ax.grid(axis='x', linestyle='--', alpha=0.6) 

# Thêm nhãn dữ liệu (data labels) cho từng thanh
ax.bar_label(bars, fmt='%.3f', padding=5, fontsize=DATA_LABEL_FONT) 

# Thêm cỡ chữ cho các nhãn trên 2 trục
ax.tick_params(axis='x', labelsize=TICK_LABEL_FONT) 
ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT) 

# Đặt giới hạn cho trục X để dễ đọc hơn
ax.set_xlim(0, df_temp_08['eodds_gap'].max() * 1.15) # Tăng 15% để có chỗ cho nhãn

plt.tight_layout()
plt.savefig('fairness-prms/evaluation_output/eodds_gap_distribution_at_temp_08.png')

print("Biểu đồ phân bổ EOdds Gap với phông chữ LỚN HƠN và Tiếng Việt đã được lưu.")