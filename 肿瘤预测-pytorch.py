import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 下载和加载数据集
data = load_breast_cancer()
X = data.data  # 特征：肿瘤尺寸等
y = data.target  # 标签：0（良性）或 1（恶性）

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# 4. 构建神经网络模型
class TumorPredictor(nn.Module):
    def __init__(self):
        super(TumorPredictor, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 16)  # 输入层到隐藏层
        self.fc2 = nn.Linear(16, 8)                  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(8, 2)                   # 隐藏层到输出层
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# 5. 初始化模型、损失函数和优化器
model = TumorPredictor()
criterion = nn.NLLLoss()  # 使用负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
epochs = 1000
fval_history = np.zeros((0))
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    fval_history = np.append(fval_history, [loss.item()], axis=0)
# 7. 预测
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred_tensor, dim=1).numpy()  # 获取预测类别

# 8. 评估模型
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'模型的准确率: {accuracy:.2f}')

#loss
plt.plot(fval_history)
plt.title('Figure of Loss')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.savefig(fr'ann_train.png')
plt.show()

# 9. 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_classes))
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 10. 混淆矩阵可视化
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['良性', '恶性'], yticklabels=['良性', '恶性'])
plt.title('混淆矩阵')
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.show()