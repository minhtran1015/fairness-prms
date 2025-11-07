#!/usr/bin/env python3
"""
Script to plot category data with Vietnamese labels as a Lollipop Chart
(Version with easy-to-adjust label settings)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Dữ liệu và Cài đặt của bạn ---

# Font settings
AXIS_LABEL_FONT = 23
TICK_LABEL_FONT = 18
DATA_LABEL_FONT = 18

# === BẢNG ĐIỀU KHIỂN CĂN CHỈNH LABEL ===
# Thay đổi 3 giá trị này để căn chỉnh nhãn
LABEL_PADDING_OFFSET = 400  # Khoảng cách từ điểm tròn tới nhãn (tính bằng đơn vị)
X_AXIS_LIMIT_FACTOR = 1.2  # % không gian trống (1.2 = 20% trống)
# ------------------------------------

# Data (Giữ nguyên)
categories_en = [
    'Age', 'Disability_status', 'Gender_identity', 'Nationality', 
    'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', 
    'Race_x_SES', 'Religion', 'SES', 'Sexual_orientation'
]
values = [3680, 1556, 5672, 3080, 1576, 6880, 15960, 11160, 1200, 6864, 864]
translation_map = {
    "Disability_status": "Tình trạng khuyết tật", "SES": "Tình trạng KT-XH",
    "Race_x_gender": "Chủng tộc & Giới tính", "Age": "Tuổi tác",
    "Race_ethnicity": "Chủng tộc & Sắc tộc", "Race_x_SES": "Chủng tộc & KT-XH",
    "Gender_identity": "Bản dạng giới", "Religion": "Tôn giáo",
    "Nationality": "Quốc tịch", "Sexual_orientation": "Xu hướng tính dục",
    "Physical_appearance": "Ngoại hình"
}
categories_vi = [translation_map[cat] for cat in categories_en]
sorted_indices = np.argsort(values)[::-1]
categories_vi_sorted = [categories_vi[i] for i in sorted_indices]
values_sorted = [values[i] for i in sorted_indices]

# --- HÀM VẼ BIỂU ĐỒ (ĐÃ CẬP NHẬT) ---

def create_lollipop_chart():
    """Create a horizontal lollipop chart with Vietnamese labels"""
    
    fig, ax = plt.subplots(figsize=(10, 12)) 
    y_pos = np.arange(len(categories_vi_sorted))
    colors = sns.color_palette("Set2", len(categories_vi_sorted))
    labels_formatted = [f'{v:,}' for v in values_sorted]

    for i in range(len(categories_vi_sorted)):
        ax.hlines(y=y_pos[i], xmin=0, xmax=values_sorted[i], 
                  color=colors[i], alpha=0.7, linewidth=3)
        
        ax.plot(values_sorted[i], y_pos[i], "o", 
                markersize=10, color=colors[i], alpha=0.9)
        
        # === THAY ĐỔI 1: Sử dụng biến padding ===
        ax.text(values_sorted[i] + LABEL_PADDING_OFFSET, # Vị trí X
                y_pos[i],              # Vị trí Y
                labels_formatted[i],   # Nhãn (đã định dạng)
                va='center',
                fontsize=DATA_LABEL_FONT)

    # --- Tùy chỉnh biểu đồ ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories_vi_sorted, fontsize=TICK_LABEL_FONT)
    ax.set_xlabel('Số lượng', fontsize=AXIS_LABEL_FONT)
    ax.set_ylabel('Loại danh mục', fontsize=AXIS_LABEL_FONT)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    
    # === THAY ĐỔI 2: Sử dụng biến giới hạn trục ===
    ax.set_xlim(0, max(values_sorted) * X_AXIS_LIMIT_FACTOR) 
    
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONT)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT)
    
    plt.tight_layout()
    return fig, ax

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    
    print("Creating horizontal lollipop chart...")
    fig_lollipop, ax_lollipop = create_lollipop_chart()
    plt.savefig('fairness-prms/evaluation_output/category_data_lollipop.png', dpi=300, bbox_inches='tight')
    plt.close(fig_lollipop)
    
    print("\nBiểu đồ Lollipop đã được lưu thành công!")