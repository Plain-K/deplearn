import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# 2. 将数据保存到CSV文件
data.to_csv('Wine_Quality_Data.csv', index=False)

# 3. 数据预处理
X = data.drop('quality', axis=1)  # 特征
y = data['quality']                # 标签

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 创建决策树分类器
dt_classifier = DecisionTreeClassifier(random_state=42)

# 6. 训练决策树模型
dt_classifier.fit(X_train, y_train)

# 7. 预测
y_pred_dt = dt_classifier.predict(X_test)

# 8. 评估决策树模型
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'决策树模型的准确率: {accuracy_dt:.2f}')

# 9. 输出决策树分类报告
print("\n决策树分类报告:")
print(classification_report(y_test, y_pred_dt))

# 10. 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, 
                                       max_depth=3, min_samples_split=2, 
                                       min_samples_leaf=1, max_features='sqrt', 
                                       max_leaf_nodes=10, min_impurity_decrease=0.01, 
                                       bootstrap=True, oob_score=True, n_jobs=-1, 
                                       class_weight='balanced', ccp_alpha=0.01, 
                                       max_samples=None)

# 11. 训练随机森林模型
rf_classifier.fit(X_train, y_train)

# 12. 预测
y_pred_rf = rf_classifier.predict(X_test)

# 13. 评估随机森林模型
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'随机森林模型的准确率: {accuracy_rf:.2f}')

# 14. 输出随机森林分类报告
print("\n随机森林分类报告:")
print(classification_report(y_test, y_pred_rf))

# 15. 混淆矩阵可视化
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(12, 6))

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制随机森林混淆矩阵
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', 
            annot_kws={"size": 16}, 
            xticklabels=[3, 4, 5, 6, 7, 8], 
            yticklabels=[3, 4, 5, 6, 7, 8])
plt.title('随机森林混淆矩阵', fontsize=16)
plt.ylabel('实际值', fontsize=14)
plt.xlabel('预测值', fontsize=14)

# 绘制决策树混淆矩阵
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', 
            annot_kws={"size": 16}, 
            xticklabels=[3, 4, 5, 6, 7, 8], 
            yticklabels=[3, 4, 5, 6, 7, 8])
plt.title('决策树混淆矩阵', fontsize=16)
plt.ylabel('实际值', fontsize=14)
plt.xlabel('预测值', fontsize=14)

plt.tight_layout()
plt.show()