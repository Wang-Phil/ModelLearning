#引入必要的包
#数据处理的包
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import sklearn.datasets
#搭建网络的包
import torch.nn as nn
import torch.nn.functional as F

#生成200个点的数据集，返回的结果是一个包含两个元素的元组 (X, y),X是点，y是x的分类
X, y =  sklearn.datasets.make_moons(200, noise=0.2)
print(X,y)
