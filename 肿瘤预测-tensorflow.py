from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers


# 1. 下载和加载数据集
data = load_breast_cancer()
X = data.data  # 特征：肿瘤尺寸等
y = data.target  # 标签：0（良性）或 1（恶性）

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 构建神经网络模型
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 输出层，使用sigmoid激活函数
])

# 4. 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 6. 预测
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # 将概率转换为类别

# 7. 评估模型
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'模型的准确率: {accuracy:.2f}')

# 8. 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_classes))

# 9. 混淆矩阵可视化
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['良性', '恶性'], yticklabels=['良性', '恶性'])
plt.title('混淆矩阵')
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.show()