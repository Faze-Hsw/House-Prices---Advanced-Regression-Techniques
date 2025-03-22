import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

#读取训练数据与测试数据,并进行数据合并
df1=pd.read_csv(r"C:\Users\TS.1989\Desktop\kaggle_house_pred_train.csv")
df1=df1.set_index('Id',append=False)
df2=pd.read_csv(r'C:\Users\TS.1989\Desktop\kaggle_house_pred_test.csv')
df2=df2.set_index('Id',append=False)
df=pd.concat([df1,df2],axis=0,join='outer')

#删除训练集中缺失值比例50%以上的特征列
na_count=df1.isna().mean(axis=0)
df=df.loc[:,na_count<0.5]

#分离特征列与标签列
features=df.drop(columns='SalePrice')
labels=df[['SalePrice']]

#分别选取连续型特征和分类型特征
numerical_features=features.select_dtypes(exclude='object').columns.to_list()
classification_features=features.select_dtypes(include='object').columns.to_list()

#观察分类型特征的分布情况并记录类别不平衡情况
def diturbution_of_classification_features(classification_features):
    for feature in classification_features:
        value=features[feature].value_counts()
        plt.title(feature)
        plt.bar(value.index,value.values)
        plt.savefig(fr"C:\Users\TS.1989\Desktop\分类特征分布柱状图\{feature}.png")
        plt.clf()
diturbution_of_classification_features(classification_features)
anomaly_features=['Street',
                  'Utilities',
                  'Condition2',
                  'RoofMatl',
                  'Heating',
                  ]
features=features.drop(columns=anomaly_features) #删除类别不平衡的特征列

#观察连续型特征的分布情况并计算偏度
continuous_features=features.select_dtypes(include='float').columns.to_list()
error_features=['BsmtFinSF2','BsmtFullBath','BsmtHalfBath','GarageCars']
continuous_features=[x for x in continuous_features if x not in error_features]
print(continuous_features)
def distribution_of_continuous_features(continuous_features):
    for feature in continuous_features:
        value=features[feature].values
        plt.title(feature)
        plt.hist(value,bins=50)
        plt.savefig(fr"C:\Users\TS.1989\Desktop\连续特征直方图\{feature}.png")
        plt.clf()
distribution_of_continuous_features(continuous_features)
# 统计偏度大于1的列
skew=features[continuous_features].loc[1:1460].apply(lambda x:x.skew())
#使用box-cox转换处理偏度绝对值大于1的特征列
for feature in skew.index:
    if abs(skew[feature])>1:
        transformer=PowerTransformer(method='box-cox')
        features.loc[1:1460,[feature]]=transformer.fit_transform(features.loc[1:1460,[feature]].values+1).reshape(-1)
        features.loc[1461:,[feature]]=transformer.transform(features.loc[1461:,[feature]].values+1).reshape(-1)
#保存转换后的连续特征分布直方图
for feature in continuous_features:
    value=features[feature].values
    plt.title(feature)
    plt.hist(value,bins=50)
    plt.savefig(fr'C:\Users\TS.1989\Desktop\转换后连续特征直方图\{feature}.png')
    plt.clf()

#独热编码分类特征
features=pd.get_dummies(features,dummy_na=True)
features=features.replace({False:0,True:1})

#进行Z-Score标准化
mean=features.loc[1:1460,numerical_features].mean(axis=0)
std=features.loc[1:1460,numerical_features].std(axis=0)
features[numerical_features]=(features[numerical_features]-mean)/std

#用均值填充缺失值
features[numerical_features]=features[numerical_features].fillna(value=0)

#利用方差膨胀因子判断共线性,并删除VIF大于5的列
vif_data = pd.DataFrame()
vif_data["feature"] = features.loc[1:1460,numerical_features].columns
vif_data["VIF"] = [variance_inflation_factor(features.loc[1:1460,numerical_features].values, i) for i in range(features.loc[1:1460,numerical_features].shape[1])]
print("方差膨胀因子（VIF）：")
print(vif_data.sort_values(by="VIF", ascending=False))
delete_columns=['2ndFlrSF','GrLivArea','LowQualFinSF','1stFlrSF','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','GarageCars','GarageArea','YearBuilt']
features=features.drop(columns=delete_columns)

#合并处理后的特征列与标签列，并保存结果
data=pd.concat([features,labels],axis=1)
data.to_csv(r"C:\Users\TS.1989\Desktop\data.csv")




