import numpy as np
import Pretreatment
import matplotlib.pyplot as plt
import KNN
import WKNN
import PTM_KNN
import PTM_WKNN
import NEW_PTM_WKNN
import pandas as pd
from decimal import Decimal
np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 求取正类
def get_positive_class(matrix):
    # 输入：矩阵（带标签）
    # 输出：正类
    class_1 = 0
    class_0 = 0
    for row in range(matrix.shape[0]):
        if int(matrix[row][-1]) == 2:
            class_1 += 1
        elif int(matrix[row][-1]) != 2:
            class_0 += 1
        else:
            return print('error')
    class1 = ('正类', '反类')
    class1_number = [class_1,class_0]
    return class1,class1_number


def autolabel(rects):  # 自动缩进宽度
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

# 数据集
def main():
    data_set = np.loadtxt('haberman.csv', dtype=np.float, delimiter=',', skiprows=1)
    best_k = int(KNN.get_KNNbestK(data_set))
    recallknn, fscoreknn, gmeanknn,TP, FN, TN, FP = KNN.KNN(data_set, best_k)
    recallwknn, fscorewknn, gmeanwknn,TP1, FN1, TN1, FP1 = WKNN.WKNN(data_set, best_k)
    recallpknn, fscorepknn, gmeanpknn,TP2, FN2, TN2, FP2 = PTM_KNN.PTM_KNN(data_set)
    recallpwknn, fscorepwknn, gmeanpwknn,TP3, FN3, TN3, FP3 = PTM_WKNN.PTM_WKNN(data_set)
    recallnpwknn, fscorenpwknn, gmeannpwknn, TP4, FN4, TN4, FP4 = NEW_PTM_WKNN.NEW_PTM_WKNN(data_set)
    print(TP, FN, TN, FP)
    print(TP1, FN1, TN1, FP1)
    print(TP2, FN2, TN2, FP2)
    print(TP3, FN3, TN3, FP3)
    print(TP4, FN4, TN4, FP4)
    class1, class1_number = get_positive_class(data_set)
    rects = plt.barh(class1, class1_number)
    plt.title('正反类分布')
    for rect in rects:
        width = rect.get_width()
        plt.text(width, rect.get_y() + rect.get_height() / 2, str(width), ha='center', va='bottom')
    plt.show()

    buy_number = [int(recallknn*10000)/100, int(recallwknn*10000)/100, int(recallpknn*10000)/100, int(recallpwknn*10000)/100, int(recallnpwknn*10000)/100]
    buy_number2 = [int(fscoreknn*10000)/100, int(fscorewknn*10000)/100, int(fscorepknn*10000)/100, int(fscorepwknn*10000)/100, int(fscorenpwknn*10000)/100]
    buy_number3 = [int(gmeanknn*10000)/100, int(gmeanwknn*10000)/100, int(gmeanpknn*10000)/100, int(gmeanpwknn*10000)/100, int(gmeannpwknn*10000)/100]
    name = ['KNN', 'WKNN', 'PTM-KNN', 'PTM-WKNN', 'NEW_PTM-WKNN']
    total_width, n = 2, 3
    width = total_width / n
    x = [0, 2.5, 5, 7.5, 10]
    a = plt.bar(x, buy_number, width=width, label='Recall', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    b = plt.bar(x, buy_number2, width=width, label='F-score', tick_label=name, fc='r')
    for i in range(len(x)):
        x[i] = x[i] + width
    c = plt.bar(x, buy_number3, width=width, label='G-mean', fc='b')
    autolabel(a)
    autolabel(b)
    autolabel(c)
#
    plt.xlabel('算法')
    plt.ylabel('百分比（%）')
    plt.title('实验结果')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()