import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. 下载和加载数据集
data = load_breast_cancer()
X = data.data  # 特征：肿瘤尺寸等
y = data.target  # 标签：0（良性）或 1（恶性）

# 保存为CSV文件
df = pd.DataFrame(data=np.c_[X, y], 
                  columns=data.feature_names.tolist() + ['target'])
df.to_csv('breast_cancer_data.csv', index=False)
# 2. 数据预处理
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 实例化LDA模型
lda = LinearDiscriminantAnalysis()

# 6. 训练模型
lda.fit(X_train, y_train)

# 7. 测试模型
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

