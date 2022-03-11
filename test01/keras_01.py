import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
# import matplotlib.pyplot as plt # 可视化模块


# create some data
filename = 'H:/Py/VLC/VLC_20190512(PyTorch)/Data/20190521.csv'
names = ['A', 'B', 'C', 'X', 'Y']
data = pd.read_csv(filename, names=names, delim_whitespace=True)
array = data.values
intensity_00 = np.array(array[:, 0:3], dtype=np.float32)
X_and_Y = np.array(array[:, 3:5], dtype=np.float32)
# length = len(intensity_00)
min_max_scaler = preprocessing.MinMaxScaler((-1, 1), 1)
intensity_01 = min_max_scaler.fit_transform(intensity_00)

X = intensity_01
Y = X_and_Y
# print(X[200:250])


model = Sequential()
model.add(Dense(output_dim=2, input_dim=3))
model.compile(loss='mse', optimizer='sgd')
for step in range(3000):
    cost = model.train_on_batch(X, Y)

# save
print('test before save: ', model.predict(X[200:206]))
model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model

# load
model = load_model('my_model.h5')
print('test after load: ', model.predict(X[200:206]))
"""
# save and load weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
# save and load fresh network without trained weights
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
"""
