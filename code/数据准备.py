import pandas as pd
from sklearn.preprocessing import PowerTransformer,StandardScaler
from sklearn.decomposition import PCA

#读取数据并划分特征与标签
data=pd.read_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\data.csv")
data=data.set_index('Id',append=False)
x=data.drop(columns='SalePrice')
y=data.loc[1:1460,['SalePrice']]

#统计数值型变量及分类型变量
numerical_columns=x.select_dtypes(include='number').columns.to_list()
category_columns=x.select_dtypes(include='object').columns.to_list()

#去除重复值
data=data.drop_duplicates()

#缺失值处理
x[numerical_columns]=x[numerical_columns].fillna(value=x[numerical_columns].mean(axis=0))
x[category_columns]=x[category_columns].fillna(value=x[category_columns].mode(axis=0).loc[0,:])

#极值处理
skewness=x[numerical_columns].skew(axis=0)
print(skewness)
problem_columns=['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea','OpenPorchSF']
transformer=PowerTransformer(method='box-cox')
x.loc[1:1460,problem_columns]=transformer.fit_transform(x.loc[1:1460,problem_columns]+1)
x.loc[1461:,problem_columns]=transformer.transform(x.loc[1461:,problem_columns]+1)
skewness_processed=x[numerical_columns].skew(axis=0)
print(skewness_processed)

#标准化
scaler=StandardScaler()
x.loc[1:1460,numerical_columns]=scaler.fit_transform(x.loc[1:1460,numerical_columns])
x.loc[1461:,numerical_columns]=scaler.transform(x.loc[1461:,numerical_columns])

#分类变量独热编码
x=pd.get_dummies(x,dummy_na=False)
x=x.replace([True,False],[1.0,0.0])

#PCA主成分分析
pca=PCA(n_components=15)
trans_values=pca.fit_transform(x[numerical_columns].values)
print(sum(pca.explained_variance_ratio_))
trans_data=pd.DataFrame(data=trans_values,index=x.index)
x=pd.concat((x,trans_data),axis=1)
x=x.drop(columns=numerical_columns)

#生成完成清洗之后的训练数据集与测试数据集
data_processed=pd.concat((x,y),axis=1)
train_processed=data_processed.loc[1:1460,:]
test_processed=data_processed.loc[1461:,:].drop(columns='SalePrice')
train_processed.to_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\train_processed.csv")
test_processed.to_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\test_processed.csv")


