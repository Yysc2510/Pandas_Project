import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns

# 假设data_encoded是经过预处理的数据集，X和y分别是特征和目标变量

# 1. 描述性统计分析扩展
print(data_encoded.describe())
print(data_encoded.corr()['Price'].sort_values(ascending=False))

# 2. 特征的统计检验
# 示例：对'Region'特征进行ANOVA检验
region_groups = [data_encoded[data_encoded['Region'] == region]['Price'] for region in data_encoded['Region'].unique()]
fvalue, pvalue = stats.f_oneway(*region_groups)
print("F-value:", fvalue)
print("P-value:", pvalue)

# 3. 特征工程
data_encoded['Living_Space_per_Floor'] = data_encoded['Size'] / data_encoded['Floor']

# 4. 异常值检测与处理
Q1 = data_encoded['Price'].quantile(0.25)
Q3 = data_encoded['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_encoded['Price_Outlier'] = (data_encoded['Price'] < lower_bound) | (data_encoded['Price'] > upper_bound)

# 5. 数据转换
data_encoded['Price_Log'] = np.log1p(data_encoded['Price'])

# 6. 特征选择
model = RandomForestRegressor()
model.fit(X, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature importances:")
for f in range(5):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]}")

# 7. 聚类分析
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)
print("Cluster labels:", clusters)

# 8. 残差分析
residuals = y - model.predict(X)
plt.hist(residuals, bins=30, alpha=0.7)
plt.title('Residuals Distribution')
plt.show()

# 9. 时间序列分析
data_encoded['Date'] = pd.to_datetime(data_encoded['Date'])
data_encoded['Year'] = data_encoded['Date'].dt.year
data_encoded['Month'] = data_encoded['Date'].dt.month
yearly_avg_price = data_encoded.groupby('Year')['Price'].mean().reset_index()
print(yearly_avg_price)

# 10. 数据平衡（示例使用SMOTE）
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_encoded, y)

# 可视化部分
plt.figure(figsize=(10, 8))
sns.heatmap(data_encoded.corr(), annot=True, fmt=".2f")
plt.show()

# 特征重要性可视化
features = np.array(X.columns)
plt.barh(range(len(features)), importances, align='center')
plt.yticks(range(len(features)), features)
plt.show()