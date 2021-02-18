import distance
import numpy as np
import Pretreatment
import analysis
# 加权KNN算法(WKNN)
def weighted_knn(unknown_instance, matrix, k, normalized_test_matrix):
    # 第一步：求得欧式距离，将其存入e_distance矩阵中
    e_distance = distance.e_distance_calculation(normalized_test_matrix[unknown_instance, :-1], matrix[:, :-1])  # 求取欧氏距离矩阵
    label_vector = matrix[:, -1]  # 标签向量
    labeled_distance_matrix = np.column_stack((e_distance, label_vector))  # 带有标签的距离矩阵
    sorted_distance_matrix = labeled_distance_matrix[np.argsort(labeled_distance_matrix[:, 0])]  # 对带标签向量的距离矩阵进行升序排序
    # k值由参数传入
    k_neighbours_feature = sorted_distance_matrix[:k, :]  # k最近邻详细信息
    k_neighbours_vector = sorted_distance_matrix[:k, -1]  # k最近邻标签向量
    k_distance = k_neighbours_feature[:, :-1]  # k最近邻的距离
    weight_matrix = np.ones(k_distance.shape, dtype=np.float)  # 单位矩阵
    weight_matrix = weight_matrix / k_distance  # k最近邻的权值
    labeled_weighted_k_neighbours = np.column_stack((weight_matrix, k_neighbours_vector))  # k最近邻权重标签矩阵
    # 对结果进行求取
    # 选取距离最近的k个近邻点，统计其所在类别出现的次数
    class_count = {}  # 定义字典存储不同类别的权值
    for i in range(k):
        dict_key = labeled_weighted_k_neighbours[i][1]
        # 类别不存在键值赋0，存在则加权
        class_count[dict_key] = class_count.get(dict_key, 0) + labeled_weighted_k_neighbours[i][0]
        # 使用字典.get()方法统计次数(键值)，值若不存在赋0
    # print('{类别 : 权值}:', class_count)
    # 多数表决，输出结果
    max_key = 0  # 设置一个最大字典键，空
    max_value = -1  # 设置一个最大键值，-1
    for key, value in class_count.items():  # 遍历字典找出最大的键值对
        if value > max_value:
            max_key = key
            max_value = value
    return max_key


def WKNN(data_set,best_k):
    train, test = Pretreatment.partitioningTPMKNN(data_set)
    normalized_train_matrix = Pretreatment.normalization(train)
    normalized_test_matrix = Pretreatment.normalization(test)
    predictions = np.zeros((normalized_test_matrix.shape[0], 1))
    for unknowid_text in range(0, len(normalized_test_matrix)):
        predictions[unknowid_text] = weighted_knn(unknowid_text, normalized_train_matrix, int(best_k), normalized_test_matrix)
    TP, FP, FN, TN = analysis.TPFPFNTN(normalized_test_matrix, predictions)
    return analysis.recall(TP, FN), analysis.F_score(TP, FP, FN), analysis.G_mean(TP, FN, TN, FP),TP, FN, TN, FP