import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

# 一些宏定义
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])    
BATCH_SIZE = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32

# 读取数据
script_dir = pathlib.Path(__file__).parent
print(script_dir)
train_data = script_dir / 'data/train/'
val_data = script_dir / 'data/val/'

# 数据预处理
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)   
train_data_gen = image_generator.flow_from_directory(directory=str(train_data),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,#打乱数据顺序
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=list(CLASS_NAMES)
                                                     )
val_data_gen = image_generator.flow_from_directory(directory=str(val_data),
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                   classes=list(CLASS_NAMES)
                                                   )
print("训练集样本数:", train_data_gen.samples)
print("验证集样本数:", val_data_gen.samples)

# 搭建LeNet模型
model = keras.Sequential()
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=120, kernel_size=5, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

# 模型编译
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# 模型训练
history = model.fit(train_data_gen,epochs=100,validation_data=val_data_gen)
model.save('lenet.h5')

# 绘制训练和验证的准确率
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()