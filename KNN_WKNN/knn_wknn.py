import numpy as np


def e_distance_calculation(unknown_instance, feature_matrix):
    """
    求取欧氏距离
        input: 待测点、属性的特征矩阵（不带标签向量）
        output: 欧氏距离矩阵
    :param unknown_instance: 未知样本
    :param feature_matrix: 特征矩阵
    :return distance_matrix: 欧氏距离（NumPy矩阵）
    """

    # 定义distance_matrix维度
    distance_matrix = np.zeros((feature_matrix.shape[0], 1))

    for i in range(0, len(feature_matrix)):
        distance_matrix[i] = float(np.sqrt(np.sum(np.square(feature_matrix[i] - unknown_instance))))

    return distance_matrix


def normalization(matrix):
    """
    矩阵归一化
    :param matrix: 未进行归一化的矩阵
    :return normalized: 归一化后的矩阵
    """

    label_vector = matrix[:, -1]
    max_element = matrix[:, :-1].max()
    min_element = matrix[:, :-1].min()

    normalized = (matrix[:, :-1] - min_element) / (max_element - min_element)
    normalized = np.column_stack((normalized, label_vector))

    return normalized


def partition(full_set):
    """
    数据集划分
    :param full_set: 原始数据集
    :return train_set: 训练集
    :return test_set: 测试集
    """

    np.random.seed(3)  # 指定随机数种子

    full_set = np.unique(full_set, axis=0)  # 删除数据集重复样本
    shuffled_matrix = np.random.permutation(full_set)  # 打乱数据集

    row = shuffled_matrix.shape[0]  # 获取总样本数

    train_set = shuffled_matrix[:int(row * 0.7), :]  # 训练集前70%
    test_set = shuffled_matrix[int(row * 0.7):, :]  # 测试集后30%

    return train_set, test_set


def pretreatment(dataset):
    """
    数据预处理
        Input: 数据集
        Output: 归一化后且划分好的训练集和测试集
    :param dataset: 数据集
    :return train_set: 训练集
    :return test_set: 测试集
    """

    # 归一化
    normalized = normalization(dataset)

    # 数据集划分
    train_set, test_set = partition(normalized)

    return train_set, test_set


def traditional_knn(unknown_instance, k, normalized_matrix):
    """
    传统KNN算法
        Input: 未知实例、未进行向量组切分的训练集矩阵、归一化后的训练集分类标签向量
        Output: 预测结果
    :param unknown_instance: 未知样本
    :param k: k值
    :param normalized_matrix: 归一化后的测试集矩阵
    :return max_key: 预测值
    """

    # 求取欧氏距离矩阵
    e_distance = e_distance_calculation(unknown_instance, normalized_matrix[:, :-1])

    # 获取标签向量
    label_vector = normalized_matrix[:, -1]

    # 将距离矩阵和标签向量组合成一个矩阵
    labeled_distance_matrix = np.column_stack((e_distance, label_vector))

    # 对带标签向量的距离矩阵进行升序排序
    sorted_distance_matrix = labeled_distance_matrix[np.argsort(labeled_distance_matrix[:, 0])]

    # 选取距离最近的k个近邻点，统计其所在类别出现的次数（使用字典方式）

    class_count = {}  # 定义类别出现次数字典

    for i in range(k):
        # 循环取排好序后的前k个样本的标签
        dict_key = sorted_distance_matrix[i][-1]

        # 使用字典.get()方法统计次数(键值)，值若不存在赋0
        class_count[dict_key] = class_count.get(dict_key, 0) + 1

    max_key = 0  # 设置一个最大字典键，空
    max_value = -1  # 设置一个最大键值，-1

    for key, value in class_count.items():  # 遍历字典找出最大的键值对
        if value > max_value:
            max_key = key
            max_value = value

    return max_key


def weighted_knn(unknown_instance, k, normalized_matrix):
    """
    加权KNN算法
        Input: 未知实例、未进行向量组切分的训练集矩阵、归一化后的训练集分类标签向量
        Output: 预测结果
    :param unknown_instance: 未知样本
    :param k: k值
    :param normalized_matrix: 归一化后的测试集矩阵
    :return max_key: 预测值
    """

    # 求取欧氏距离矩阵，通常加权的权值为欧氏距离的倒数
    e_distance = 1 / (e_distance_calculation(unknown_instance, normalized_matrix[:, :-1]))

    # 获取标签向量
    label_vector = normalized_matrix[:, -1]

    # 将距离矩阵和标签向量组合成一个矩阵
    labeled_distance_matrix = np.column_stack((e_distance, label_vector))

    # 对带标签向量的距离矩阵进行升序排序
    sorted_distance_matrix = labeled_distance_matrix[np.argsort(labeled_distance_matrix[:, 0])]

    # 选取距离最近的k个近邻点，统计其所在类别出现的次数（使用字典方式）

    class_count = {}  # 定义类别出现次数字典

    for i in range(k):

        # 循环取排好序后的前k个样本的标签
        dict_key = sorted_distance_matrix[i][-1]

        # 使用字典.get()方法统计次数(键值)，值若不存在赋0
        class_count[dict_key] = class_count.get(dict_key, 0) + sorted_distance_matrix[i][0]

    max_key = 0  # 设置一个最大字典键，空
    max_value = -1  # 设置一个最大键值，-1

    for key, value in class_count.items():  # 遍历字典找出最大的键值对
        if value > max_value:
            max_key = key
            max_value = value

    return max_key


def accuracy(normalized_matrix):
    """
    计算精确度
    :param normalized_matrix: 归一化后的矩阵
    :return acc: 精确度
    """

    # 未完成，源文件丢失
    acc = normalized_matrix
    return acc


def main(dataset):
    """
    主程序（未完成，源文件丢失）
    :param dataset: 数据集
    :return:
    """

    # 传入数据
    # file = open( , r)

    # 数据预处理
    # train_matrix, test_matrix = pretreatment(file)

    # KNN进行预测
    # traditional_knn()

    # 得出结果

    # 在测试集上进行结果对比
    # traditional_knn()


if __name__ == '__main__':
    main()
