import numpy as np
import math
def e_distance_calculation(unknown_instance, feature_matrix):
    # 输入：待测点、属性的特征矩阵（不带标签向量）
    # 输出：欧氏距离矩阵
    klist = np.zeros((feature_matrix.shape[0], 1))
    for i in range(0, len(feature_matrix)):
        klist[i] = float(np.sqrt(np.sum(np.square(feature_matrix[i] - unknown_instance))))
    return klist