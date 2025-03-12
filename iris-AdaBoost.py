import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建基础分类器
base_classifier = DecisionTreeClassifier(max_depth=1)  # 使用决策树作为基础分类器

# 4. 创建AdaBoost分类器
ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)

# 5. 训练模型
ada_boost.fit(X_train, y_train)

# 6. 预测
y_pred = ada_boost.predict(X_test)

# 7. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'AdaBoost模型的准确率: {accuracy:.2f}')

# 8. 可视化特征重要性
importances = ada_boost.feature_importances_
plt.bar(range(len(importances)), importances)
plt.xlabel('index')
plt.ylabel('importance')
plt.title('AdaBoost')
plt.xticks(range(len(importances)), iris.feature_names, rotation=45)
plt.show()