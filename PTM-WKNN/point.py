import numpy as np
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def partitioningTPMKNN(full_set): #训练集，测试集分为7：3比例
    np.random.seed(8)
    shuffled_matrix = np.random.permutation(full_set)#打乱数据集
    row = shuffled_matrix.shape[0]
    train_set = shuffled_matrix[:int(row * 0.7), :]
    test_set = shuffled_matrix[int(row * 0.7):, :]
    train_set = np.unique(train_set, axis=0)  # 删除训练集重复行
    test_set = np.unique(test_set, axis=0)  # 删除测试集重复行
    return train_set, test_set


def main():

    data_set = np.loadtxt('haberman.csv', dtype=np.float, delimiter=',', skiprows=1)
    train, test = partitioningTPMKNN(data_set)
    ax = []
    ay = []
    az = []
    bx = []
    by = []
    bz = []
    ax1 = []
    ay1 = []
    az1 = []

    for i in range(0, len(train)):
        if int(train[i][3]) == 1:
            ax.append(train[i][0])
            ay.append(train[i][1])
            az.append(train[i][2])
        if int(train[i][3]) == 2:
            bx.append(train[i][0])
            by.append(train[i][1])
            bz.append(train[i][2])
    for j in range(0, len(test)):
        if int(test[j][3]) == 1 or int(test[j][3]) == 2:
            ax1.append(test[j][0])
            ay1.append(test[j][1])
            az1.append(test[j][2])


    print(ax1)
    fig = plt.figure()
    # 将二维转化为三维
    axes3d = Axes3D(fig)
    # axes3d.scatter3D(x,y,z)
    # 效果相同
    axes3d.scatter(ax, ay, az, color='#00CED1', label='已知点第1类')

    axes3d.scatter(ax1, ay1, az1, color='#000000', marker='x', label='未知点')
    plt.title('haberman数据集')
    plt.show()


if __name__ == "__main__":
    main()