import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from sklearn import 

# 数据预处理
script_dir = os.path.dirname(__file__)
print(script_dir)
original_data_path = os.path.join(script_dir, "breast_cancer_data.csv")
original_data = pd.read_csv(original_data_path)
print(original_data)

# 拆分特征与标签
x_data = original_data.iloc[:, :-1]
y_data = original_data.iloc[:, -1]
print("x_data:", x_data)
print("y_data:", y_data)

#数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
x_data_normalized = scaler.fit_transform(x_data)
print("x_data_normalized:", x_data_normalized)

# 划分数据集和验证集
x_train, x_test, y_train, y_test = train_test_split(x_data_normalized, y_data, test_size=0.2, random_state=42)#固定随机种子random_state=42
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)

# logistic回归模型训练
lr = LogisticRegression()
lr.fit(x_train, y_train)        
print("w:", lr.coef_)
print("b:", lr.intercept_)

#进行预测
y_pred = lr.predict(x_test)
print("y_pred:", y_pred)

#查看预测结果
result = pd.DataFrame({
    "真实标签": y_test.values,
    "预测结果": y_pred
})
print(result)  
#查看错误行
print(result[result["真实标签"] != result["预测结果"]])

# 模型评估
print("模型评估:")
print(classification_report(y_test, y_pred,target_names=['良性肿瘤', '恶性肿瘤']))

