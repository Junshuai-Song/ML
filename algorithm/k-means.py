# coding=utf-8
'''
Created on 2016年5月10日
@author: a1

算法名称：k-means聚类，非监督聚类方法
（1）初始数据没有区分类别，需要我们按照特征进行聚类
（2）相似性按照距离来衡量；
（3）选取k个初始质心（通常随机选取）
（4）循环至收敛：
        对每个样本，计算其与k个质心的距离最近的那个，将其归为对应的类
        重新计算质心，当质心不再变化即为收敛。(一般是每个类别的样本点平均，即质心)
        
要点：
（1）实际目标：最小化均方差
（2）聚类个数需要自己设定
（3）样本距离依据特征不同可以单独设定

测试数据集：[x1,x2,...,类别]

缺点：
（1）k-means是局部最优，对初始质心的选取敏感。
（2）选择能达到目标函数最优的k值是非常困难的。

注意：
    centroids[cent,:]=np.mean(ptsInCent,axis=0) 
    其中可能会存在初始的中心，在某一轮分类后没有任何一个样例接近它，这个分类为0，但是由于k的限定，也不能取消掉；
    可能在后序迭代中会再次用到此类别中心;
    如果最后都没有再用到，此中心会输出nan.

'''
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

def read_file(fn):
    
    dataSet=[]
    with open(fn) as f:
        raw_data = np.loadtxt(fn,delimiter=',',dtype="float",usecols=None)
    for row in raw_data:
        dataSet.append(row)

    return np.array(dataSet)


def firstCentroids(k,dataSet):
    """
        创建k个初始质心,k行，有几个属性就有几列。
    """
    num_columns=dataSet.shape[1]    # 读取矩阵第2维长度，这里表示有列属性
    centroids=np.zeros((k,num_columns)) # k行、num_columns列
    
    for j in range(num_columns):
        minJ=min(dataSet[:,j])      #第j列
        rangeJ=max(dataSet[:,j])-minJ
        for i in range(k):
            centroids[i,j]=minJ+rangeJ*np.random.uniform(0,1)   # 对于每一列，生成当前列最大值最小值之间的一个数
            
    return np.array(centroids)
    
    
def k_means(k,dataSet):
    num_rows,num_columns=dataSet.shape
    centroids=firstCentroids(k, dataSet)
    
    # store the cluster that the samples belong to
    clusterAssment=np.zeros((num_rows,2))
    
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        
        # find the closet centroid
        for i in range(num_rows):
            minDis=np.Inf
            minIndex=-1
            for j in range(k):  # 对于每一个样例，需要与k和中心比较，选择距离最近的那个即是类别
                distance=ssd.euclidean(dataSet[i,:],centroids[j,:]) # 计算距离
                if distance<minDis:
                    minDis=distance; minIndex=j
            
            if(clusterAssment[i,0]!=minIndex): # 属于的类别不对
                clusterChanged=True
                clusterAssment[i,:]=minIndex,minDis**2  # 乘方
                
        # update the centroid location
        for cent in range(k):
            ptsInCent=dataSet[np.nonzero(clusterAssment[:,0]==cent)[0]]
            # print ptsInCent
            # axis表示计算每一列的和，加上mean取平均值，最为新的质心
            centroids[cent,:]=np.mean(ptsInCent,axis=0)     # 某一轮可能某类别中心样例数为0
            
    return centroids
        
def k_means_test(k):
    dataSet=read_file('/Users/a1/Desktop/ali/algorithm_data/k-means/k-means.txt')
    ans=k_means(k, dataSet)
    print k
    print ans   # 代表k个类型中心

k_means_test(3)
<<<<<<< HEAD
    
=======
    
>>>>>>> 49d456a2a175f3daca606e5d04bc97b2108e4505
