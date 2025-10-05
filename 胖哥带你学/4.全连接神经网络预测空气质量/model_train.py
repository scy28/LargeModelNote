import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')
data = pd.read_csv(data_path)
print(data)

# 数据提取和划分
x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 分别归一化
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)
y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = y_scaler.transform(y_test.values.reshape(-1,1))

# 模型建立
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# 模型编译
model.compile(optimizer='adam', loss='mse')

# 模型训练
history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(x_test, y_test))
model.save("model.h5")


# 设置全局字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()
