import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from numpy import concatenate
from math import sqrt

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')
data = pd.read_csv(data_path)
print(data)

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')
data = pd.read_csv(data_path)
print(data)

# 数据提取和划分
x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
print("真实值",y_test)
# 分别归一化
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)
y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = y_scaler.transform(y_test.values.reshape(-1,1))

# 模型加载
model = load_model("model.h5")

# 模型预测
y_pred = model.predict(x_test)
# print("预测值:", y_pred)

# 预测值反归一化
y_pred = concatenate((x_test, y_pred), axis=1)
y_pred = y_scaler.inverse_transform(y_pred)
y_pred = y_pred[:, -1]
print("反归一化后的预测值:", y_pred)


y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)
# print("真实值:", y_test)

# 真实值反归一化
y_test = concatenate((x_test, y_test), axis=1)
y_test = y_scaler.inverse_transform(y_test)
y_test = y_test[:, -1]
print("反归一化后的真实值:", y_test)

# 计算均方误差（MSE）
mse = np.mean((y_pred - y_test) ** 2)
print("均方误差（MSE）:", mse)

# 计算均方根误差（RMSE）
rmse = sqrt(mse)
print("均方根误差（RMSE）:", rmse)

# 计算平均绝对百分比误差（MAPE）
mape = np.mean(np.abs((y_pred - y_test) / y_test))
print("平均绝对百分比误差（MAPE）:", mape)


# 设置全局字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  
# 绘制预测值与真实值对比图
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="真实值")
plt.plot(y_pred, label="预测值")
plt.title("全连接神经网络预测空气质量对比图")
plt.legend()
plt.show()
