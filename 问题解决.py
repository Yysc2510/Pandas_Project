import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# 1. 数据生成与预处理
# 数据加载
data = pd.read_csv('lianjia.csv', encoding='utf8')

# 数据概览
print(data.head())

# 数据预处理
# 处理缺失值（简单起见，我们直接删除有缺失值的行）
data.dropna(inplace=True)

# 编码分类特征
categorical_features = ['Direction', 'District', 'Elevator', 'Layout', 'Region', 'Renovation']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(data[categorical_features])

# 转换成 DataFrame 并添加回原始数据中
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
data_encoded = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)
data_encoded.drop(columns=categorical_features, inplace=True)

# 选取特征和目标变量
X = data_encoded.drop(columns=['Price', 'Id', 'Garden'])
y = data_encoded['Price']

# 处理并标准化数值特征（假设 'Floor' 列是数值格式）
X['Floor'] = X['Floor'].astype(int)  # 如果 'Floor' 列是字符串，需要提前转换为整数
num_features = ['Floor', 'Size', 'Year']
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# 2. 数据分析
# 数据概述
print(data.describe())

# 类别特征的值频率
for feature in categorical_features:
    print(f'\n值计数 {feature}:')
    print(data[feature].value_counts())

# 切分房屋面积
size_bins = pd.cut(data['Size'], bins=[0, 50, 100, 150, 200, np.inf], labels=['0-50', '50-100', '100-150', '150-200', '200+'])
print('\n箱的大小计数:')
print(size_bins.value_counts())

# 按区域分组统计平均价格
region_group = data.groupby('Region')['Price'].mean().reset_index()
print('\n按地区划分的平均价格:')
print(region_group)

# 按装修情况分组统计平均价格
renovation_group = data.groupby('Renovation')['Price'].mean().reset_index()
print('\n装修平均价格:')
print(renovation_group)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型搭建与运行
# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"均方根误差: {rmse}")

# 打印模型系数
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# 4. 数据可视化
plt.figure(figsize=(16, 8))

# 原始数据加载用于可视化
original_data = pd.read_csv('lianjia.csv', encoding='utf8').dropna()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用你的字体路径
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 价格分布
plt.subplot(2, 3, 1)
original_data['Price'].plot(kind='hist', bins=30, alpha=0.7, ax=plt.gca())
plt.title('价格分布')

# 大小和价格关系
plt.subplot(2, 3, 2)
original_data.plot(kind='scatter', x='Size', y='Price', alpha=0.7, ax=plt.gca())
plt.title('大小与价格')

# 楼层和价格关系
plt.subplot(2, 3, 3)
original_data.plot(kind='scatter', x='Floor', y='Price', alpha=0.7, ax=plt.gca())
plt.title('楼层与价格')

# 各区房价分布
plt.subplot(2, 3, 4)
original_data.boxplot(column='Price', by='Region', ax=plt.gca())
plt.title('各地区价格分布')
plt.suptitle('')  # 移除默认的子标题
plt.xticks(rotation=90)

# 年份和价格关系
plt.subplot(2, 3, 5)
original_data.plot(kind='scatter', x='Year', y='Price', alpha=0.7, ax=plt.gca())
plt.title('年份与价格')

# 装修和价格关系
plt.subplot(2, 3, 6)
original_data.boxplot(column='Price', by='Renovation', ax=plt.gca())
plt.title('装修价格分布')
plt.suptitle('')  # 移除默认的子标题

plt.tight_layout()
plt.show()

# 5. 输入预测价格
# 预测函数
def predict_price(input_data):
    # 编码和标准化输入数据
    input_df = pd.DataFrame([input_data])
    encoded_input = encoder.transform(input_df[categorical_features])
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())
    input_df = pd.concat([input_df.reset_index(drop=True), encoded_input_df], axis=1)
    input_df.drop(columns=categorical_features, inplace=True)

    # 标准化数值特征
    input_df[num_features] = scaler.transform(input_df[num_features])

    # 预测
    return model.predict(input_df)[0]

# 用户输入数据进行预测
input_data = {
    'Direction': '南北',
    'District': '东单',
    'Elevator': '无电梯',
    'Floor': '6',
    'Layout': '2室1厅',
    'Region': '东城',
    'Renovation': '精装',
    'Size': 60,
    'Year': 1988
}

predicted_price = predict_price(input_data)
print(f"输入数据的预测价格为： {predicted_price}")
