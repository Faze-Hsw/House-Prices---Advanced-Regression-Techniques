import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import init
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error

device=torch.device('cuda')
#读取清洗过后的数据集,并划分测试集训练集
df=pd.read_csv(r"C:\Users\TS.1989\Desktop\data.csv")
df=df.set_index('Id',append=False)
x=df.drop(columns='SalePrice').loc[1:1460,:].values
x=torch.tensor(x).float()
x=x.to(device)
validate=df.drop(columns='SalePrice').loc[1461:,:].values
validate=torch.tensor(validate).float()
validate=validate.to(device)
y=df.loc[1:1460,'SalePrice'].values
y=torch.tensor(y).float().reshape(-1,1)
y=y.to(device)

#划分训练集与测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,test_size=0.1,random_state=42,shuffle=True)

#创建训练数据迭代器与测试数据迭代器:
dataset_train=data.TensorDataset(x_train,y_train)
train_iter=data.DataLoader(dataset_train,batch_size=64)
dataset_test=data.TensorDataset(x_test,y_test)
test_iter=data.DataLoader(dataset_test,batch_size=64)

#创建损失计算器
def evaluate_loss(data_iter,net):
    net.eval()
    total_loss=0
    total_len=0
    for u,v in data_iter:
        y_pred=net(u).reshape(-1).detach().to('cpu')
        y_true=v.reshape(-1).detach().to('cpu')
        total_loss+=root_mean_squared_log_error(y_true,y_pred)
        total_len+=1
    return total_loss/total_len

#创建神经网络训练器并实现训练损失可视化
def train(net,loss,epochs,data_iter,lr):
    best_loss = float('inf')
    patience = 20
    counter = 0
    #创建优化器
    trainer=torch.optim.Adam(net.parameters(),lr,weight_decay=1)
    #创建记录每次训练的损失表
    step=[i for i in range(1,epochs+1)]
    step_count=0
    loss_train=[]
    loss_test=[]
    #训练并记录损失
    for epoch in range(1,epochs+1):
        #将神经网络设定为训练模式
        net.train()
        step_count+=1
        for u,v in data_iter:
            l=loss(net(u),v)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        train_loss=evaluate_loss(train_iter,net)
        test_loss=evaluate_loss(test_iter,net)
        print(f'epoch:{epoch},loss:{train_loss}')
        loss_train.append(train_loss)
        loss_test.append(test_loss)
        # 早停逻辑
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    #可视化训练过程中的损失
    plt.plot(step[0:step_count],loss_train,label='train_loss')
    plt.plot(step[0:step_count],loss_test,label='test_loss')
    plt.yscale('log')
    plt.legend()
    plt.show()
    print(evaluate_loss(test_iter,net)) #打印最终的测试集RMSE

#创建多层感知机神经网络
net = nn.Sequential(
    nn.Linear(268, 512),nn.ReLU(),nn.Dropout(p=0.3),
    nn.Linear(512, 1)
)
net.to(device)

#参数初始化
for layer in net:
    if isinstance(layer,nn.Linear):
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)

#创建损失函数
loss=nn.MSELoss()

#训练模型
train(net,loss,2000,train_iter,lr=0.001)

#用神经网络预测并保存结果
net.eval()
price_prediction=net(validate).reshape(-1).detach()
price_prediction=price_prediction.to('cpu')
result=pd.DataFrame({'Id':[i for i in range(1461,2920)],
                     'SalePrice':price_prediction})
result.to_csv(r"C:\Users\TS.1989\Desktop\results.csv",index=False)