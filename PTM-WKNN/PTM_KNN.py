import distance
import numpy as np
import Pretreatment
import analysis
import KNN


def testlocal_k(normalized_train_matrix, normalized_test_matrix, unknowid, localc):  # 获取第i个测试集的localk值
    unknow_instance = normalized_test_matrix[unknowid, :-1]  # 未知样例
    unknow_distance = distance.e_distance_calculation(unknow_instance, normalized_train_matrix[:, :-2])
    label_vector = normalized_train_matrix[:, -1]
    labeled_distance = np.column_stack((unknow_distance, label_vector))  # 距离与对应k标签合并
    sorted_labeled_distance = labeled_distance[labeled_distance[:, 0].argsort()]  # 排序
    max = -1
    for c in range(0, localc):
        if sorted_labeled_distance[c][1] > max:
            max = sorted_labeled_distance[c][1]
    return max

def trainlocal_k(normalized_train_matrix, unknowid):  # 获取第i个训练集的localk值
    apart_train_matrix = normalized_train_matrix
    apart_train_matrix = np.delete(apart_train_matrix, unknowid, 0)
    local_k_list = []
    local_k = 1
    for i in range(1, 20, 2):
        local_k_list.append(i)
    for ki in local_k_list:
        answer = KNN.traditional_knn(unknowid, apart_train_matrix, ki, normalized_train_matrix)  # 计算knn的预测分类结果
        if answer == normalized_train_matrix[unknowid][-1]:  # 判断预测结果和真实结果是否一致
            local_k = ki
            break
    return local_k


def PTM_KNN(data_set):
    train, test = Pretreatment.partitioningTPMKNN(data_set)
    normalized_train_matrix = Pretreatment.normalization(train)
    normalized_test_matrix = Pretreatment.normalization(test)
    klist = np.zeros((normalized_train_matrix.shape[0], 1))
    predictions = np.zeros((normalized_test_matrix.shape[0], 1))
    for unknowid in range(0, len(normalized_train_matrix)):
        klist[unknowid] = trainlocal_k(normalized_train_matrix, unknowid)
    new_normalized_train_matrix = np.column_stack((normalized_train_matrix, klist))
    for unknowid_text in range(0, len(normalized_test_matrix)):
        new_k = testlocal_k(new_normalized_train_matrix, normalized_test_matrix, unknowid_text, 3)
        predictions[unknowid_text] = KNN.traditional_knn(unknowid_text, normalized_train_matrix, int(new_k), normalized_test_matrix)
    TP, FP, FN, TN = analysis.TPFPFNTN(normalized_test_matrix, predictions)
    return analysis.recall(TP, FN), analysis.F_score(TP, FP, FN), analysis.G_mean(TP, FN, TN, FP),TP, FN, TN, FP