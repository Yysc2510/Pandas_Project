# 1. 数据生成与预处理
data = pd.read_csv('D:\\python\\lianjia.csv', encoding='gbk')
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
X['Floor'] = X['Floor'].astype(int)  
num_features = ['Floor', 'Size', 'Year']
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])
