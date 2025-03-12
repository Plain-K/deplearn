import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1. 加载Wine数据集
wine = load_wine()
X = wine.data
y = wine.target

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 测试不同的n_neighbors值并进行交叉验证
neighbors = range(1, 101)
mean_accuracies = []

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn, X_train, y_train, cv=10)  # 10折交叉验证
    mean_accuracies.append(scores.mean())

# 4. 绘制准确率与n_neighbors的关系
plt.plot(neighbors, mean_accuracies)
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Accuracy')
plt.title('KNN Mean Accuracy vs. Number of Neighbors (Cross-Validation)')
plt.xticks(range(0, 101, 10))  # 每10个点显示
plt.grid()
plt.show()

# 5. 输出最优K值
optimal_k = neighbors[np.argmax(mean_accuracies)]
print(f'最优K值: {optimal_k}')