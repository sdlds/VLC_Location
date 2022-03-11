'''
PyTorch神经网络线性拟合可见光数据
调用 matplotlib 进行绘图分析
加入 save() restore_params() 模块

运行程序前注意更改各文件保存加载名
JunqiangZhang@tom.com
200191105
'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# 3, 400, 2, 200000, 0.01
# 超参数设定
feature = 3
hidden = 100
output = 2
EPOCH = 10000
LR = 0.0001

# 载入数据
filename = 'H:\\Py\\VLC\\VLC_20191112(PyTorch)\\Data\\4.csv'
loss_name = 'H:\\Py\\VLC\\VLC_20191112(PyTorch)\\Data\\shice20191109(1).txt'
loss_name2 = 'H:\\Py\\VLC\\VLC_20191112(PyTorch)\\Data\\shice20191109(2).txt'
loss_name3 = 'H:\\Py\\VLC\\VLC_20191112(PyTorch)\\Data\\shice20191109(3).txt'
parameters_name = 'shice1109(1).pkl'
names = ['A', 'B', 'C', 'X', 'Y']
# data = pd.read_csv(filename, names=names, delim_whitespace=True)
data = pd.read_csv(filename, names=names, sep = ',')

data2 = pd.read_csv('H:\\Py\\VLC\\VLC_20191112(PyTorch)\\Data\\20191110.csv', names=names, sep =',')
# 分离数据
array = data.values       # pandas to numpy
array2 = data2.values       # pandas to numpy
# print(type(array))
print(len(array))
intensity_00 = np.array(array[:, 0:3], dtype=np.float32)
coordinates = np.array(array[:, 3:5], dtype=np.float32)

intensity_test02 = np.array(array2[:, 0:3], dtype=np.float32)
coordinates_test02 = np.array(array2[:, 3:5], dtype=np.float32)
# print(intensity_00[:, 1])
seed = 2019

min_max_scaler = preprocessing.MinMaxScaler((0, 1), 1)
intensity_01 = min_max_scaler.fit_transform(intensity_00)
intensity_02 = min_max_scaler.fit_transform(intensity_test02)

# intensity_train, intensity_test, coordinates_train, coordinates_test = \
#     train_test_split(intensity_01, coordinates, test_size=0, random_state=seed)
intensity_train = intensity_01
coordinates_train = coordinates

intensity_test = intensity_02
coordinates_test = coordinates_test02
length = len(intensity_train)
length_test = len(intensity_test)
# print(length)
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(3, hidden),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden, 2)
)

def save():
    Loss_list = []
    Loss_list2 = []
    optimizer = torch.optim.Adam(net_dropped.parameters(), lr=LR, betas=(0.9, 0.99))
    # criterion = nn.MSELoss()

    for epoch in range(EPOCH):
        # Convert numpy array to torch Variable
        inputs = torch.from_numpy(intensity_train)
        targets = torch.from_numpy(coordinates_train)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        pred = net_dropped(inputs)
        loss = torch.sum(((torch.sum(((pred - targets)**2), 1, True))**(1/2)))/length
        loss.backward()
        optimizer.step()
        save_loss = loss.detach().numpy()
        Loss_list.append(save_loss)
        inputs_test1 = torch.from_numpy(intensity_test)
        targets_test1 = torch.from_numpy(coordinates_test)
        pred = net_dropped(inputs_test1)
        # print('###################################################################')
        # print(pred)
        error = torch.sum((torch.sum(((pred - targets_test1) ** 2), 1, True)) ** (1 / 2)) / length
        error_n = error.detach().numpy()
        Loss_list2.append(error_n)
        if (epoch + 1) % 500 == 0:
            print('Epoch [%d/%d], Loss: %.4f'
                  % (epoch + 1, EPOCH, loss.item()))
    # save the net
    torch.save(net_dropped, 'model.h5')
    # net_dropped.dump()
    # torch.save(net_dropped.state_dict(), parameters_name)  # save only the parameters
    ## plot
    x1 = range(0, EPOCH)
    y1 = Loss_list
    filename = open(loss_name, 'w')
    # for value in Loss_list:
    #     filename.write(str(value)+'\n')
    # filename.close()
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.xlabel('train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.show()
    filename = open(loss_name2, 'w')
    # for value in Loss_list2:
    #     filename.write(str(value)+'\n')
    # filename.close()

def restore_params():
    # criterion = nn.MSELoss()
    Loss_list3 = []
    net_dropped.load_state_dict(torch.load(parameters_name))
    inputs_test1 = torch.from_numpy(intensity_test)
    targets_test1 = torch.from_numpy(coordinates_test)
    pred = net_dropped(inputs_test1)
    print('###################################################################')
    # print(pred)
    error = ((torch.sum(((pred - targets_test1) ** 2), 1, True)) ** (1 / 2))
    error_n = error.detach().numpy()
    Loss_list3.append(error_n)
    filename = open(loss_name3, 'w')
    for value in Loss_list3:
        filename.write(str(value)+'\n')
    filename.close()
    # for i in range(length_test):
    #     print(error_n[i])
    # print('###################################################################')
    print(length_test)

save()
# restore_params()
