import numpy as np  
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
# 数据准备
filename = 'H:\\Py\\VLC\\VLC_20191112(PyTorch)\\Data\\20191107.csv'
names = ['A', 'B', 'C', 'X', 'Y']
data = pd.read_csv(filename, names=names, delim_whitespace=True)
#分离数据
array = data.values
intensity_00 = array[:, 0:3]
X_Y = array[:, 3:5]
# X = array[:, 3]
# Y = array[:, 4]
min_max_scaler = preprocessing.MinMaxScaler((0, 1), 1)
intensity_01 = min_max_scaler.fit_transform(intensity_00)
validation_size = 0  # 样本占比如果是整数的话就是样本的数量
seed = 2019   # 随机数的种子,填0或不填每次都会不一样
intensity_train, intensity_test, X_Y_train, X_Y_test = train_test_split(intensity_01, X_Y, test_size=validation_size, random_state=seed)
X_train = X_Y_train[:, 0]
Y_train = X_Y_train[:, 1]
X_test = X_Y_test[:, 0]
Y_test = X_Y_test[:, 1]

n_folds = 10 
model_br = BayesianRidge() 
model_lr = LinearRegression()  
model_etc = ElasticNet()  
model_svr = SVR()  
model_gbr = GradientBoostingRegressor()  
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR'] 
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  
cv_score_list = []  
pre_y_list = []  
for model in model_dic:  
    scores = cross_val_score(model, intensity_01, Y, cv=n_folds)  
    cv_score_list.append(scores) 
    pre_y_list.append(model.fit(intensity_01, Y).predict(intensity_01))  
n_samples, n_features = intensity_01.shape  
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  
model_metrics_list = []  
for i in range(5):  
    tmp_list = []  
    for m in model_metrics_name:  
        tmp_score = m(Y, pre_y_list[i]) 
        tmp_list.append(tmp_score) 
    model_metrics_list.append(tmp_list)  
df1 = pd.DataFrame(cv_score_list, index=model_names) 
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  
#############################################################################################################
# with open('H:/Py/VLC_190331/serial_code/save/model_X.mexw64', 'wb') as f01:
#     pickle.dump(model_gbr, f01)
with open('H:/Py/VLC_190331/serial_code/save/model_Y.mexw64', 'wb') as f02:
   pickle.dump(model_gbr, f02)
#############################################################################################################
print ('samples: %d \t features: %d' % (n_samples, n_features))
print (70 * '-')
print ('cross validation result:')
print (df1)
print (70 * '-')
print ('regression metrics:')
print (df2)
print (70 * '-')
print ('short name \t full name')
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  
plt.figure() 
plt.plot(np.arange(intensity_01.shape[0]), Y, color='k', label='true y01') 
color_list = ['r', 'b', 'g', 'y', 'c']  
linestyle_list = ['-', '.', 'o', 'v', '*']  
for i, pre_y in enumerate(pre_y_list):  
    plt.plot(np.arange(intensity_01.shape[0]), pre_y_list[i], color_list[i], label=model_names[i]) 
plt.title('regression result comparison') 
plt.legend(loc='upper right')  
plt.ylabel('real and predicted value')  
plt.show()  
print ('regression prediction')
# a = 304
# b = 439
# c = 986
intensity_p = np.array([[304, 439, 986]])
intensity_new01 = np.concatenate((intensity_00, intensity_p), axis=0)
##############################################
min_max_scaler = preprocessing.MinMaxScaler((0, 1), 1)
intensity_train_minmax = min_max_scaler.fit_transform(intensity_new01)
length = len(intensity_00)
end_intensity = intensity_train_minmax[[length]]
for i, new_point in enumerate(end_intensity): 
    new_pre_y0 = model_br.predict((new_point.reshape(1, -1))) # reshape(a, b) a行b列 b=-1时，自动计算列数
    new_pre_y1 = model_lr.predict((new_point.reshape(1, -1)))
    new_pre_y2 = model_etc.predict((new_point.reshape(1, -1)))
    new_pre_y3 = model_svr.predict((new_point.reshape(1, -1)))
    new_pre_y4 = model_gbr.predict((new_point.reshape(1, -1)))
    print('%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (new_pre_y0,new_pre_y1,new_pre_y2,new_pre_y3,new_pre_y4))
    #print('%.2f' % (new_pre_y4)) 

print(model_br.score(intensity_01, Y))
print(model_lr.score(intensity_01, Y))
print(model_etc.score(intensity_01, Y))
print(model_svr.score(intensity_01, Y))
print(model_gbr.score(intensity_01, Y))
