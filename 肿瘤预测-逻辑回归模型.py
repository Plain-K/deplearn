import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# 1. 下载和加载数据集
data = load_breast_cancer()
X = data.data  # 特征：肿瘤尺寸等
y = data.target  # 标签：0（良性）或 1（恶性）

# 2. 保存为CSV文件
df = pd.DataFrame(data=np.c_[X, y], 
                  columns=data.feature_names.tolist() + ['target'])
df.to_csv('breast_cancer_data.csv', index=False)

# 3. 数据预处理
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 实例化逻辑回归模型
model = LogisticRegression(max_iter=1000)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 测试模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')