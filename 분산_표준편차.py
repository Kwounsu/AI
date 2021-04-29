import numpy as np

np.random.seed(0)
data_A = np.random.randint(0, 100, 10000)

data_A_var = np.var(data_A)
data_A_std = np.std(data_A)

print(f'분산 : {data_A_var.round(2)}')
print(f'표준편차 : {data_A_std.round(2)}')
# 분산 : 833.0
# 표준편차 : 28.86
