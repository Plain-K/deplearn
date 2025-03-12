import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. 加载手写数字数据集
#digits = datasets.load_digits()
digits = datasets.fetch_olivetti_faces()
X = digits.data
y = digits.target

# 2. 使用StratifiedShuffleSplit确保每个类别都有样本
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 3. 定义不同的核函数
kernels = ['linear', 'rbf', 'poly']
accuracies = []
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 4. 训练和评估每个核函数的SVM模型
for kernel in kernels:
    svm_classifier = SVC(kernel=kernel, random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f'核函数: {kernel}, 准确率: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    # 混淆矩阵可视化
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'混淆矩阵 - 核函数: {kernel}')
    plt.xlabel('预测值')
    plt.ylabel('实际值')
    plt.show()

# 5. 绘制准确率比较
plt.figure(figsize=(8, 5))
plt.bar(kernels, accuracies, color='skyblue')
plt.xlabel('核函数')
plt.ylabel('准确率')
plt.title('不同核函数的SVM分类准确率比较')
plt.ylim(0, 1)
plt.show()