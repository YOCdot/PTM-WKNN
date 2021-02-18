import distance
import numpy as np
import Pretreatment
import WKNN
import analysis
import PTM_KNN


def bestc(normalized_train_matrix, normalized_test_matrix, unknowid, data_set):
    unknow_instance = normalized_test_matrix[unknowid, :-1]  # 未知样例
    unknow_distance = distance.e_distance_calculation(unknow_instance, normalized_train_matrix[:, :-2])
    label_vector = normalized_train_matrix[:, -1]
    labeled_distance = np.column_stack((unknow_distance, label_vector))  # 距离与对应k标签合并
    sorted_labeled_distance = labeled_distance[labeled_distance[:, 0].argsort()]  # 排序
    max = -1
    real_accuracy = 0
    for cnumber in range(1, 3):  # 循环取c值
        for c in range(0, cnumber):  # 循环选取周围c个数中最大的标签k值
            if sorted_labeled_distance[c][1] > max:
                max = sorted_labeled_distance[c][1]
        a = 0
        for k in range(1, 11):  # 十折交叉验证
            train, test = Pretreatment.partitioning(data_set, k)  # 十折拆分数据集
            normalized_train_matrix = Pretreatment.normalization(train)
            normalized_test_matrix = Pretreatment.normalization(test)
            predictions = np.zeros((normalized_test_matrix.shape[0], 1))  # 初始化预测结果集合
            for unknowid_text in range(0, len(normalized_test_matrix)):
                predictions[unknowid_text] = WKNN.weighted_knn(unknowid_text, normalized_train_matrix, int(max), normalized_test_matrix)
            accuracy = analysis.getAccuracy(normalized_test_matrix, predictions)
            a = accuracy + a
        thistest_accuracy = float(a / 10)
        if thistest_accuracy > real_accuracy: # 保存十折交叉验证后精度最高的c值
            real_accuracy = thistest_accuracy
            best_c = cnumber
    return best_c


def NEW_PTM_WKNN(data_set):
    train, test = Pretreatment.partitioningTPMKNN(data_set)
    normalized_train_matrix = Pretreatment.normalization(train)
    normalized_test_matrix = Pretreatment.normalization(test)
    klist = np.zeros((normalized_train_matrix.shape[0], 1))
    predictions = np.zeros((normalized_test_matrix.shape[0], 1))
    for unknowid in range(0, len(normalized_train_matrix)):  # 给训练集样本加入标签local k
        klist[unknowid] = PTM_KNN.trainlocal_k(normalized_train_matrix, unknowid)
    new_normalized_train_matrix = np.column_stack((normalized_train_matrix, klist))
    for unknowid_text in range(0, len(normalized_test_matrix)):  # 选取测试集样本最近三个点的最大local k作为最佳k
        best_c = bestc(new_normalized_train_matrix, normalized_test_matrix, unknowid_text, data_set) # 循环选取c值
        new_k1 = PTM_KNN.testlocal_k(new_normalized_train_matrix, normalized_test_matrix, unknowid_text, best_c)  #通过c值选取k值
        predictions[unknowid_text] = WKNN.weighted_knn(unknowid_text, normalized_train_matrix, int(new_k1), normalized_test_matrix)
        # 计算邻近k个值中分类权重最高的分类作为预测分类
    TP, FP, FN, TN = analysis.TPFPFNTN(normalized_test_matrix, predictions)
    return analysis.recall(TP, FN), analysis.F_score(TP, FP, FN), analysis.G_mean(TP, FN, TN, FP),TP, FN, TN, FP