import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 資料
years = [2020, 2021, 2022, 2023, 2024]
speeds = [19, 130, 620, 4000, 20000]
prices = [1000, 140, 40, 13, 3]

# 設定圖表大小
fig, ax1 = plt.subplots(figsize=(10, 6))

# 繪製價格的長條圖
ax1.bar(years, prices, color='b', alpha=0.6, label='Price')
ax1.set_xlabel('Year')
ax1.set_ylabel('Price', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 使用相同的x軸，但新的y軸繪製速度的折線圖
ax2 = ax1.twinx()
ax2.plot(years, speeds, color='r', marker='o', linestyle='-', label='Speed')
ax2.set_ylabel('Speed', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 添加標題
plt.title('GPU Price and Speed (2020-2024)')

# 添加圖例
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# 顯示圖表
plt.show()
