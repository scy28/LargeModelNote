import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import cv2

# 一些宏定义
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])    
IMG_HEIGHT = 32
IMG_WIDTH = 32

# 读取数据
img = cv2.imread("data/val/Cr/Cr_48.bmp")
plt.imshow(img)
plt.show()
img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
img = img / 255.0

# 使用模型
model = keras.models.load_model("lenet.h5")
# 缺陷检测
pred = model.predict(img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3))
print(pred)
print(CLASS_NAMES[np.argmax(pred)])
