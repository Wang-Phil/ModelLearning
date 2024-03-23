#引入必要的包
#数据处理的包
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import sklearn.datasets
#搭建网络的包
import torch
import torch.nn as nn
import torch.nn.functional as F

#生成200个点的数据集，返回的结果是一个包含两个元素的元组 (X, y),X是点，y是x的分类
X, y =  sklearn.datasets.make_moons(200, noise=0.2)

save_path = "/data/wangweicheng/ModelLearning/SimpleNetWork"
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)
plt.savefig(f'{save_path}/dataset.png')     #保存生成的图片

# 将 NumPy 数组转换为 PyTorch 张量，并指定张量的数据类型
X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)


# 搭建网络
class BinaryClassifier(nn.Module):
    #初始化，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
    def __init__(self,n_feature,n_hidden,n_output):
        super(BinaryClassifier, self).__init__()
        #输入层到隐藏层的全连接
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        #隐藏层到输出层的全连接
        self.output = torch.nn.Linear(n_hidden, n_output)

    #前向传播，把各个模块连接起来，就形成了一个网络结构
    def forward(self, x):
        x = self.hidden(x)  
        x = torch.tanh(x)   #激活函数
        x = self.output(x)
        return x
    

     #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)


#初始化模型
model = BinaryClassifier(2, 3, 2)
#定义损失函数，用于衡量模型预测结果与真实标签之间的差异。
loss_criterion = nn.CrossEntropyLoss()
#定义优化器，优化器（optimizer）用于更新模型的参数，以最小化损失函数并提高模型的性能
# model.parameters() 表示要优化的模型参数，lr=0.01 表示学习率（learning rate）为 0.01，即每次参数更新的步长。
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#训练的次数
epochs = 10
#存储loss
losses = []

for i in range(epochs):
    #得到预测值
    y_pred = model.forward(X)
    #计算当前的损失
    loss = loss_criterion(y_pred, y)
    #添加当前的损失到losses中
    losses.append(loss.item())
    #清楚之前的梯度
    optimizer.zero_grad()
    #反向传播更新参数
    loss.backward()
    #梯度优化
    optimizer.step()


from sklearn.metrics import accuracy_score
print(accuracy_score(model.predict(X),y))