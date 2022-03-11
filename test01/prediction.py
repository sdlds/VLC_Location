'''
PyTorch神经网络线性拟合可见光数据
加入 save() restore_params() 模块
save() 进行训练保存模型参数
restore_params() 加载参数进行预测
JunqiangZhang@tom.com
200190514
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn import preprocessing

filename = 'H:/Py/VLC/VLC_20190512(PyTorch)/Data/20190521.csv'
names = ['A', 'B', 'C', 'X', 'Y']
data = pd.read_csv(filename, names=names, delim_whitespace=True)
array = data.values
intensity_00 = np.array(array[:, 0:3], dtype=np.float32)
X_and_Y = np.array(array[:, 3:5], dtype=np.float32)
length = len(intensity_00)
min_max_scaler = preprocessing.MinMaxScaler((-1, 1), 1)
intensity_01 = min_max_scaler.fit_transform(intensity_00)



feature = 3
hidden = 100
output = 2
num_epochs = 100000
learning_rate = 0.01

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        out = self.predict(x)
        return out

def save():
    net1 = Net(n_feature=feature, n_hidden=hidden, n_output=output)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net1.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(intensity_01))
        targets = Variable(torch.from_numpy(X_and_Y))
        optimizer.zero_grad()
        pred = net1(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            print('Epoch [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, loss.item()))
    torch.save(net1.state_dict(), 'net_params4.pkl')
    print('The model has been successfully generated!')

def restore_params(a, b, c):
    intensity_new0 = np.array([[a, b, c]])
    intensity_new0 = intensity_new0.astype(np.float32)
    intensity_new1 = np.concatenate((intensity_00, intensity_new0), axis=0)
    min_max_scaler = preprocessing.MinMaxScaler((-1, 1), 1)
    intensity_train_minmax = min_max_scaler.fit_transform(intensity_new1)
    intensity_new2 = intensity_train_minmax[[length]]
    intensity_new3 = torch.from_numpy(intensity_new2)

    net2 = Net(n_feature=feature, n_hidden=hidden, n_output=output)
    net2.load_state_dict(torch.load('net_params4.pkl'))
    prediction = net2(Variable(intensity_new3)).data.numpy()
    print(prediction)
    data_x = prediction[0, 0]
    data_y = prediction[0, 1]
    return ([data_x, data_y])

save()