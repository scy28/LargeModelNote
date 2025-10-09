import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

#读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
data_path = os.path.join(script_dir, "breast_cancer_data.csv")
data = pd.read_csv(data_path)
print(data)

x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
print("x_data:", x_data)
print("y_data:", y_data)

#数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
x_data_normalized = scaler.fit_transform(x_data)
print("x_data_normalized:", x_data_normalized)



#划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_data_normalized, y_data, test_size=0.2, random_state=42)
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)

#将标签转换为one-hot编码
y_train_onehot = to_categorical(y_train, num_classes=2)
y_test_onehot = to_categorical(y_test, num_classes=2)
print("y_train:", y_train_onehot)
print("y_test:", y_test_onehot)

#模型建立
full_connect = keras.Sequential()
full_connect.add(Dense(units=10, activation='relu'))
full_connect.add(Dense(units=20, activation='relu'))
full_connect.add(Dense(units=32, activation='relu'))
full_connect.add(Dense(units=2, activation='softmax'))

#编译模型
full_connect.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#设置保存规则
mode_save_path = os.path.join(script_dir, "model.h5")
checkpoint = ModelCheckpoint(filepath=mode_save_path, monitor="val_loss", save_best_only=True, verbose=1)
#训练模型
history = full_connect.fit(x_train, y_train_onehot, epochs=100, batch_size=32, verbose=2, validation_data=(x_test, y_test_onehot),callbacks=[checkpoint])

# 设置全局字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()
