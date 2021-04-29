import numpy as np
from scipy import stats

np.random.seed(0)
data_A = np.random.randint(0, 100, 10000)

mean = np.mean(data_A)
median = np.median(data_A)
mode = stats.mode(data_A)

print(f'평균값 : {mean.round(2)}')
print(f'중앙값 : {median}')
print(f'최빈값 : {mode[0][0]} ({mode[1][0]})')  # 해당 수 (빈도)
# 평균값 : 49.17
# 중앙값 : 49.0
# 최빈값 : 3 (125)
# 정규분포 형태 X

data_B = np.random.normal(size=100)

mean = np.mean(data_B)
median = np.median(data_B)
mode = stats.mode(data_B)

print(f'평균값 : {mean.round(2)}')
print(f'중앙값 : {median}')
print(f'최빈값 : {mode[0][0]} ({mode[1][0]})')  # 해당 수 (빈도)
# 평균값 : -0.03
# 중앙값 : -0.1024051532970941
# 최빈값 : -2.862292431709955 (1)
# 정규분포 형태 O
