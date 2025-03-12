import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 1. 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# 2. 数据预处理
X = data.drop('quality', axis=1)  # 特征
y = data['quality']                # 标签

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. 训练模型
rf_classifier.fit(X_train, y_train)

# 6. 获取特征重要性
importances = rf_classifier.feature_importances_
feature_names = X.columns

# 7. 创建特征重要性数据框
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# 8. 按重要性排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# 9.设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 10. 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('重要性')
plt.title('特征重要性')
plt.show()

# 11. 输出特征重要性
print(feature_importance_df)