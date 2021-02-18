import numpy as np
import Pretreatment
import distance
import analysis
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 传统KNN算法
def traditional_knn(unknown_instance, matrix, k, normalized_test_matrix):
    # 输入：未知实例、未进行向量组切分的训练集矩阵、归一化后的训练集分类标签向量
    # 输出：预测结果
    e_distance = distance.e_distance_calculation(normalized_test_matrix[unknown_instance, :-1], matrix[:, :-1])  # 求取欧氏距离矩阵
    label_vector = matrix[:, -1]  # 标签向量
    labeled_distance_matrix = np.column_stack((e_distance, label_vector))  # 带有标签的距离矩阵
    sorted_distance_matrix = labeled_distance_matrix[np.argsort(labeled_distance_matrix[:, 0])]# 对带标签向量的距离矩阵进行升序排序
    # 选取距离最近的k个近邻点，统计其所在类别出现的次数
    class_count = {}  # 定义类别出现次数字典
    for i in range(k):
        dict_key = sorted_distance_matrix[i][1]
        class_count[dict_key] = class_count.get(dict_key, 0) + 1  # 使用字典.get()方法统计次数(键值)，值若不存在赋0
    max_key = 0  # 设置一个最大字典键，空
    max_value = -1  # 设置一个最大键值，-1
    for key, value in class_count.items():  # 遍历字典找出最大的键值对
        if value > max_value:
            max_key = key
            max_value = value
    return max_key
def get_KNNbestK(data_set):
    real_accuracy = 0
    beat_k = 1
    for ks in range(1, 20):  # 循环取k值
        a = 0
        for k in range(1, 11):
            train, test = Pretreatment.partitioning(data_set, k)  # 十折拆分数据集
            normalized_train_matrix = Pretreatment.normalization(train)
            normalized_test_matrix = Pretreatment.normalization(test)
            predictions = np.zeros((normalized_test_matrix.shape[0], 1))  # 初始化预测结果集合
            for unknowid_text in range(0, len(normalized_test_matrix)):
                predictions[unknowid_text] = traditional_knn(unknowid_text, normalized_train_matrix, int(ks), normalized_test_matrix)
            accuracy = analysis.getAccuracy(normalized_test_matrix, predictions)
            a = accuracy + a
        thistest_accuracy = float(a / 10)
        if thistest_accuracy > real_accuracy:
            real_accuracy = thistest_accuracy
            best_k = ks
    return best_k
def KNN(data_set,best_k):
    train, test = Pretreatment.partitioningTPMKNN(data_set) #3，7比例分配数据集
    normalized_train_matrix = Pretreatment.normalization(train) #归一化
    normalized_test_matrix = Pretreatment.normalization(test)
    predictions = np.zeros((normalized_test_matrix.shape[0], 1))
    for unknowid_text in range(0, len(normalized_test_matrix)):
        predictions[unknowid_text] = traditional_knn(unknowid_text, normalized_train_matrix, int(best_k), normalized_test_matrix)
    TP, FP, FN, TN = analysis.TPFPFNTN(normalized_test_matrix, predictions)
    return analysis.recall(TP, FN), analysis.F_score(TP, FP, FN), analysis.G_mean(TP, FN, TN, FP),TP, FN, TN, FP