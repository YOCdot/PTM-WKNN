def getAccuracy(normalized_test_matrix, predictions):
    correct = 0
    for x in range(len(normalized_test_matrix)):
        if normalized_test_matrix[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(normalized_test_matrix))) * 100.0


def TPFPFNTN(normalized_test_matrix, predictions):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for x in range(len(normalized_test_matrix)):
        if normalized_test_matrix[x][-1] == predictions[x] and int(normalized_test_matrix[x][-1]) == 2:
            TP += 1
        elif normalized_test_matrix[x][-1] == predictions[x] and int(normalized_test_matrix[x][-1]) != 2:
            TN += 1
        elif normalized_test_matrix[x][-1] != predictions[x] and int(normalized_test_matrix[x][-1]) == 2:
            FN += 1
        elif normalized_test_matrix[x][-1] != predictions[x] and int(normalized_test_matrix[x][-1]) != 2:
            FP += 1
    return TP, TN, FN, FP


def recall(TP, FN):
    return TP / (TP + FN)


def precision(TP, FP):
    return TP / (TP + FP)


def F_score(TP, FP, FN):
    return 2 * recall(TP, FN) * precision(TP, FP) / (recall(TP, FN) + precision(TP, FP))


def G_mean(TP, FN, TN, FP):
    return ((TP / (FN + TP)) * (TN / (FP + TN))) ** 0.5