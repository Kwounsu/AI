import pandas as pd
import matplotlib.pyplot as plt


store_a = pd.Series([20, 21, 23, 22, 26, 28, 35, 35, 40, 46, 47, 47, 48, 49, 57, 59, 70, 69, 57, 66])
store_b = pd.Series([3, 5, 6, 7, 12, 14, 15, 20, 21, 23, 22, 26, 28, 35, 35, 20, 23, 34, 45, 52, 500])

A_Q1 = store_a.quantile(0.25)  # 1사분위수: 27.5
A_Q2 = store_a.quantile(0.50)  # 2사분위수: 46.5 (중앙값)
A_Q3 = store_a.quantile(0.75)  # 3사분위수: 57.0
A_Q4 = store_a.quantile(1)     # 4사분위수: 70.0

B_Q1 = store_b.quantile(0.25)  # 1사분위수: 14.0
B_Q2 = store_b.quantile(0.50)  # 2사분위수: 22.0 (중앙값)
B_Q3 = store_b.quantile(0.75)  # 3사분위수: 34.0
B_Q4 = store_b.quantile(1)     # 4사분위수: 500.0


# Boxplot
plt.boxplot(store_a)
plt.grid()
plt.show()

plt.boxplot(store_b)
plt.grid()
plt.show()


# IRQ
box_nums = pd.Series([16, 21, 22, 23, 24, 25, 30])

Q1 = box_nums.quantile(0.25)
Q2 = box_nums.quantile(0.50)
Q3 = box_nums.quantile(0.75)
Q4 = box_nums.quantile(1)

IRQ = Q3 - Q1             # 3.0
STEP1 = IRQ * 1.5         # 4.5
lower_fence = Q1 - STEP1  # 17.0 (이 값보다 작으면 이상치)
upper_fence = Q3 + STEP1  # 29.0 (이 값보다 높으면 이상치)
