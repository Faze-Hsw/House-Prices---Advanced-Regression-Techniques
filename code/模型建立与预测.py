import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import root_mean_squared_log_error
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
#读取训练数据集与测试数据集
train=pd.read_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\train_processed.csv")
train=train.set_index('Id',append=False)
x=train.drop(columns='SalePrice').values
y=train[['SalePrice']].values.reshape(-1)

#建立Catboost模型并进行超参数调优
def search_cat():
    param_grid = {'depth': [4, 6, 8],  # 树深度
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'l2_leaf_reg': [1, 3, 5],       # L2正则化系数
    }
    model=CatBoostRegressor(iterations=1000)
    grid_search=GridSearchCV(model,param_grid,scoring='neg_root_mean_squared_log_error',cv=5,n_jobs=-1)
    grid_search.fit(x,y)
    print(grid_search.best_score_)
    return grid_search.best_estimator_
model=CatBoostRegressor(depth=5,learning_rate=0.01,l2_leaf_reg=2,iterations=2000)
print(-1*cross_val_score(model,x,y,cv=5,scoring='neg_root_mean_squared_log_error'))
#model.fit(x,y)


#读取测试数据集
test=pd.read_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\test_processed.csv")
id_list=test['Id'].values
x_test=test.drop(columns='Id').values
y_pred=model.predict(x_test)
results=pd.DataFrame({'Id':id_list,'SalePrice':y_pred})
results.to_csv(r"C:\Users\TS.1989\Desktop\House Prices - Advanced Regression Techniques\data\results.csv",index=False)