
# train_ratio = 0.8  # 训练集所占比例
# val_ratio = 0.1  # 验证集所占比例
# test_ratio = 0.1  # 测试集所占比例
# #整合数据集，并实现划分
# dataset = TensorDataset(X, y)
# total_samples = len(dataset)
# train_samples = int(train_ratio * total_samples)
# val_samples = int(val_ratio * total_samples)
# test_samples = totalimport torch
# import pandas as pd #导入pandas库以读取数据集csv文件
# import torch.nn as nn   #神经网络
# from torch.utils.data import random_split   #训练集的划分库
# from torch.utils.data import TensorDataset  #将特征与标签结合成完整数据集的库
# df = pd.read_csv('chapter_linear-network_9/boston.csv')
# #print(df.head())   #去掉注释以观察数据集的前几行
# X = df.iloc[:, :-1].values  #所有行和除去最后一列外的所有列
# X = torch.tensor(X) #将读取到的数据（numpy数组形式）转化为可以被pytorch处理的张量
# y = df.iloc[:, -1].values   #所有行和最后一列   也就是说默认的数据集都是这么存放特征和标签的
# y = torch.tensor(y)
# #定义均值和标准差，使用torch自带的归一化函数来处理特征
# mean = torch.mean(X)
# std = torch.std(X)
# X = (X - mean) / std
# #划分训练集参数_samples - train_samples - val_samples
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_samples, val_samples, test_samples])
# #划分完毕后分离X和y
# X_train = train_dataset[:][0]
# y_train = train_dataset[:][1]
# X_test = test_dataset[:][0]
# y_test = test_dataset[:][1]
# X_val = val_dataset[:][0]
# y_val = val_dataset[:][1]
# #使用神经网络库中的线性模型
# model = nn.Linear(X.shape[1], 1)
# X_train = X_train.to(torch.float32)  # 将输入数据转换为 float32 类型
# model = model.to(torch.float32)  # 将模型的权重转换为 float32 类型
# #定义损失函数
# loss = nn.MSELoss()
# #定义优化器
# import torch.optim as optim
# optimizer = optim.SGD(model.parameters(), lr=0.3)
# #训练
# num_epochs = 1000
# for epoch in range(num_epochs):
#     outputs = model(X_train)
#     y_train = y_train.float().unsqueeze(1)  # 转换为浮点型并增加维度
#     y_train = y_train.to(outputs.device)  # 使用与模型输出相同的设备

#     # 或者
#     y_test = y_test.float().unsqueeze(1)  # 转换为浮点型并增加维度
#     y_test = y_test.to(outputs.device)  # 使用与模型输出相同的设备
#     l = loss(outputs, y_train)
#     optimizer.zero_grad()
#     l.backward()
#     optimizer.step()

# # 评估模型
# with torch.no_grad():
#     model.eval()
#     X_test = X_test.to(torch.float32)  # 将测试集输入数据转换为 float32 类型
#     y_test = y_test.float().unsqueeze(1)  # 转换为浮点型并增加维度
#     y_test = y_test.to(model.device)  # 使用与模型相同的设备
#     test_outputs = model(X_test)
#     test_loss = loss(test_outputs, y_test)

# print(f'Test Loss: {test_loss.item():.4f}')

import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt


# 读取数据集
df = pd.read_csv('chapter_linear-network_9/boston.csv')

# 提取特征和标签
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 转换为张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 归一化特征
mean = torch.mean(X, dim=0)
std = torch.std(X, dim=0)
X = (X - mean) / std

# 划分数据集
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

total_samples = len(X)
train_samples = int(train_ratio * total_samples)
val_samples = int(val_ratio * total_samples)
test_samples = total_samples - train_samples - val_samples

train_dataset, val_dataset, test_dataset = random_split(
    TensorDataset(X, y), [train_samples, val_samples, test_samples]
)

# 获取训练集和测试集
X_train, y_train = train_dataset[:]
X_test, y_test = test_dataset[:]

# 定义模型
model = nn.Linear(X.shape[1], 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 训练模型
train_losses = []
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    
    # 计算损失
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()