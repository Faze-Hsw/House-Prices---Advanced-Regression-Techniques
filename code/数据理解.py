import pandas as pd
from ydata_profiling import ProfileReport

#读取训练集与测试集并合并
train_set=pd.read_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\train.csv")
train_set=train_set.set_index('Id',append=False)
test_set=pd.read_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\test.csv")
test_set=test_set.set_index('Id',append=False)
data=pd.concat((train_set,test_set),axis=0)

#生成统计特征及分布报告
def generate_report(data,file_name):
    report=ProfileReport(data,minimal=False)
    report.to_file(file_name)
generate_report(data,'report.html')

#删除类别分布不平衡及冗余变量
delete_columns=['Street','LandContour','Utilities','LandSlope','Condition1','Condition2','RoofStyle','RoofMatl',
                'ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir','Electrical','BsmtHalfBath','KitchenAbvGr',
                'Functional','GarageQual','GarageCond','PavedDrive','MiscFeature','SaleType','SaleCondition','MSZoning',
                'BldgType']
data=data.drop(columns=delete_columns)

#删除零值比例过多的数值型变量
delete_columns=['LowQualFinSF','EnclosedPorch','ScreenPorch','PoolArea','MiscVal','3SsnPorch','BsmtFinSF2','MasVnrArea',
                '2ndFlrSF','WoodDeckSF']
data=data.drop(columns=delete_columns)

#删除缺失值比例过高的变量
delete_columns=['Alley','MasVnrType','PoolQC','Fence',]
data=data.drop(columns=delete_columns)

#生成统计特征及分布报告2.0
generate_report(data,'report2.html')

#分类型变量顺序编码
data['ExterQual']=data['ExterQual'].replace(['Ex','Gd','TA','Fa'],[4,3,2,1])
data['BsmtQual']=data['BsmtQual'].replace(['Ex','Gd','TA','Fa'],[4,3,2,1])
data['HeatingQC']=data['HeatingQC'].replace(['Ex','Gd','TA','Fa','Po'],[5,4,3,2,1])
data['KitchenQual']=data['KitchenQual'].replace(['Ex','Gd','TA','Fa'],[4,3,2,1])
data['FireplaceQu']=data['FireplaceQu'].replace(['Ex','Gd','TA','Fa','Po'],[5,4,3,2,1])
data['GarageFinish']=data['GarageFinish'].replace(['Fin','RFn','Unf'],[3,2,1])

#生成统计特征及分布报告3.0
generate_report(data,'report3.html')

#删除相关性低的特征变量
data=data.drop(columns=['LotConfig','MSSubClass','MoSold','YrSold'])

data.to_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\data.csv")