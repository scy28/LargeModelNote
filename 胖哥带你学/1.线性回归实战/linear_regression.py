#定义数据集
#定义特征
x_data= [1,2,3]
#定义标签
y_data=[2,4,6]

#初始化参数
w=4

#定义模型
def linear_regression(x):
    return w*x

#损失度计算
def loss_calculation(x_data,y_data):
    loss_value = 0
    for x,y in zip(x_data,y_data):
        loss_value += (linear_regression(x) - y)**2
    return loss_value/len(x_data)

#梯度计算
def gradient_calculation(x_data,y_data):
    gradient_value = 0
    for x,y in zip(x_data,y_data):
        gradient_value+=2*w*(x**2)-2*x*y
    return gradient_value/len(x_data)

#模型训练
for epoch in range(100):
    loss_value = loss_calculation(x_data,y_data)
    gradient_value = gradient_calculation(x_data,y_data)
    w= w - 0.01*gradient_value
    print("训练伦次：",epoch,"参数：",w,"损失度：",loss_value)

print("学习四小时得分：",linear_regression(4))

