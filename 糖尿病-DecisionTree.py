import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据集
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=column_names)

# 2. 将数据保存到CSV文件
data.to_csv('Pima_Indians_Diabetes_Data.csv', index=False)

# 3. 数据预处理
X = data.drop('Outcome', axis=1)  # 特征
y = data['Outcome']                # 标签

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 创建决策树分类器
dt_classifier = DecisionTreeClassifier(random_state=42)

# 6. 训练模型
dt_classifier.fit(X_train, y_train)

# 7. 预测
y_pred = dt_classifier.predict(X_test)

# 8. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'决策树模型的准确率: {accuracy:.2f}')

# 9. 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 10. 混淆矩阵可视化
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制混淆矩阵
heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       annot_kws={"size": 16},  # 增大数字的字体大小
                       xticklabels=['无糖尿病', '糖尿病'], 
                       yticklabels=['无糖尿病', '糖尿病'])

# 设置颜色条字体大小
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # 增大图例字体大小

plt.ylabel('实际值', fontsize=14)  # 增大y轴标签字体
plt.xlabel('预测值', fontsize=14)  # 增大x轴标签字体
plt.title('混淆矩阵', fontsize=16)  # 增大标题字体

# 增大坐标轴上的数字字体
plt.tick_params(axis='both', labelsize=14)  # 增大坐标轴数字字体

plt.show()