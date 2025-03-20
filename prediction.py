import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV

# 加载数据
train_data = pd.read_csv('TCGA_22.csv')
test_data = pd.read_csv(file_location)

# 选择正确的特征列和目标列（根据实际数据列数调整索引）
# 示例：假设特征为第2-32列，目标为第33列
X_train = train_data.iloc[:, 1:32]  # 调整索引范围
y_train = train_data.iloc[:, 32]    # 调整目标列索引
X_test = test_data.iloc[:, 1:32]
y_test = test_data.iloc[:, 32]

# 保存特征名
feature_names = X_train.columns.tolist()

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建一个逻辑回归分类器
logistic_classifier = LogisticRegression(max_iter=1000)

# 五折交叉验证
cv_scores = cross_val_score(logistic_classifier, X_train, y_train, cv=5, scoring='accuracy')

# 输出交叉验证结果
print("五折交叉验证准确率: ", cv_scores)
print("平均交叉验证准确率: ", cv_scores.mean())

# 训练模型
logistic_classifier.fit(X_train, y_train)

# 在测试集上做出预测
predictions = logistic_classifier.predict(X_test)

# 评估模型
print("分类报告：\n", classification_report(y_test, predictions, zero_division=0))
print("准确率：", accuracy_score(y_test, predictions))

# 获取模型的系数
coefficients = logistic_classifier.coef_[0]
intercept = logistic_classifier.intercept_

# 创建一个DataFrame来展示特征重要性
importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# 按系数大小排序
importance_df = importance_df.sort_values(by='Coefficient', ascending=False)

# 打印特征重要性
print("\n特征重要性（系数）:")
print(importance_df)

# 绘制特征重要性条形图
importance_df.plot(kind='bar', x='Feature', y='Coefficient', legend=False, figsize=(10, 8))
plt.title('Feature Importance')
plt.show()

# 输出模型公式（以字符串形式）
model_formula = "Logistic Regression Model:\n"
for i, feature in enumerate(feature_names):
    model_formula += f"{feature}: {coefficients[i]:.4f}, "
# 由于intercept是数组形式，我们需要取出其中的值
model_formula += f"Intercept: {intercept[0]:.4f}"
print(model_formula)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建Lasso回归模型
lasso = LassoCV(cv=5, random_state=0, max_iter=10000, alphas=np.logspace(-4, 4, 100)).fit(X_train_scaled, y_train)

# 得到被选中的特征的掩码
feature_mask = np.abs(lasso.coef_) > 1e-5

# 检查是否有特征被选中
if not np.any(feature_mask):
    raise ValueError("Lasso回归没有选择任何特征。请检查数据或调整Lasso的正则化参数。")

# 应用掩码到特征名
selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_mask[i]]

# 打印被选中的特征
print("Selected features:", selected_features)

# 使用选中的特征训练逻辑回归模型
X_train_selected = X_train_scaled[:, feature_mask]
X_test_selected = X_test_scaled[:, feature_mask]

logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_selected, y_train)

# 在测试集上做出预测
predictions = logistic_classifier.predict(X_test_selected)

# 评估模型
print("分类报告：\n", classification_report(y_test, predictions))
print("准确率：", accuracy_score(y_test, predictions))
