import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# 两个特征向量
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 高斯核函数的定义
def gaussian_kernel(x, y, gamma=0.5):
    diff = np.linalg.norm(x - y)**2
    return np.exp(-gamma * diff)

# 计算高斯核融合
kernel_value = gaussian_kernel(x, y)
print("高斯核融合结果:", kernel_value)

# 使用 sklearn 的 rbf_kernel
gamma = 0.5
kernel_matrix = rbf_kernel([x], [y], gamma=gamma)  # 必须以2D形式输入
print("sklearn 高斯核结果:", kernel_matrix[0, 0])