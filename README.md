# Stratified-prediction-of-risk-of-recurrence-in-thyroid-cancer
本项目使用逻辑回归和Lasso特征选择方法构建预测模型，适用于生物医学数据分析场景。读者需根据论文方法自行构建验证集进行测试（论文链接待补充）。

---

## 环境依赖
- Python 3.7+
- 依赖库：
  ```bash
  pandas >= 1.2.4
  scikit-learn >= 0.24.1
  numpy >= 1.20.1
  matplotlib >= 3.3.4
数据准备
  训练数据
文件路径：./TCGA_22.csv
格式要求：
特征列：第2-32列（共31个特征）
目标列：第33列（分类标签）
首列为样本ID（自动忽略）

验证集准备
需按照论文方法构建（方法待补充）
默认应与训练集格式一致
保存为CSV文件，路径需在代码中指定

**输出说明
  交叉验证结果
五折交叉验证准确率及平均值

模型评估
分类报告（精确度/召回率/F1值）
测试集准确率

特征分析
特征重要性排序（条形图）
Lasso筛选后的关键特征列表

模型公式
逻辑回归系数及截距项

注意事项
列索引适配
  若数据列数不同，需修改以下代码段：
# 第17-20行调整索引范围
X_train = train_data.iloc[:, 1:32]  # 特征列
y_train = train_data.iloc[:, 32]    # 目标列

路径错误处理
  测试集路径需完整（原代码第14行不完整）

特征筛选
  若Lasso未选择特征，需调整正则化参数alphas

引用（待更新）
bibtex
复制
@article{your_paper,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {Journal Name},
  year    = {2023},
  url     = {to_be_updated}
}
