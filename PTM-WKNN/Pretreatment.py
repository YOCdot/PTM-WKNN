import numpy as np
import math
np.set_printoptions(suppress=True)

def partitioningTPMKNN(full_set): #训练集，测试集分为7：3比例
    np.random.seed(8)
    shuffled_matrix = np.random.permutation(full_set)#打乱数据集
    row = shuffled_matrix.shape[0]
    train_set = shuffled_matrix[:int(row * 0.7), :]
    test_set = shuffled_matrix[int(row * 0.7):, :]
    train_set = np.unique(train_set, axis=0)  # 删除训练集重复行
    test_set = np.unique(test_set, axis=0)  # 删除测试集重复行
    return train_set, test_set


def partitioning(full_set, i):  # 取到第i折时训练集测试集分割情况
    row = full_set.shape[0]
    test_set = full_set[(i - 1) * int(row * 0.1):i * int(row * 0.1), :]
    train_set1 = full_set[0:(i - 1) * int(row * 0.1), :]
    train_set2 = full_set[i * int(row * 0.1):, :]
    train_set = np.vstack((train_set1, train_set2))
    return train_set, test_set


def normalization(matrix):  # 归一化
    label_vector = matrix[:, -1]
    max_element = matrix[:, :-1].max()
    min_element = matrix[:, :-1].min()
    normalized_matrix = (matrix[:, :-1] - min_element) / (max_element - min_element)
    normalized_matrix = np.column_stack((normalized_matrix, label_vector))
    return normalized_matrix