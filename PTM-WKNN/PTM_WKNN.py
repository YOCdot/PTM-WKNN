import numpy as np
import Pretreatment
import WKNN
import analysis
import PTM_KNN


def PTM_WKNN(data_set):
    train, test = Pretreatment.partitioningTPMKNN(data_set)
    normalized_train_matrix = Pretreatment.normalization(train)
    normalized_test_matrix = Pretreatment.normalization(test)
    klist = np.zeros((normalized_train_matrix.shape[0], 1))
    predictions = np.zeros((normalized_test_matrix.shape[0], 1))
    for unknowid in range(0, len(normalized_train_matrix)):  # 给训练集样本加入标签local k
        klist[unknowid] = PTM_KNN.trainlocal_k(normalized_train_matrix, unknowid)
    new_normalized_train_matrix = np.column_stack((normalized_train_matrix, klist))
    for unknowid_text in range(0, len(normalized_test_matrix)):  # 选取测试集样本最近三个点的最大local k作为最佳k
        new_k = PTM_KNN.testlocal_k(new_normalized_train_matrix, normalized_test_matrix, unknowid_text, 3)
        predictions[unknowid_text] = WKNN.weighted_knn(unknowid_text, normalized_train_matrix, int(new_k), normalized_test_matrix)
        # 计算邻近k个值中分类权重最高的分类作为预测分类
    TP, FP, FN, TN = analysis.TPFPFNTN(normalized_test_matrix, predictions)
    return analysis.recall(TP, FN), analysis.F_score(TP, FP, FN), analysis.G_mean(TP, FN, TN, FP),TP, FN, TN, FP