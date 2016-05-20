# coding=utf-8
'''
Created on 2016年5月10日
@author: a1

算法名称：KNN分类算法：监督分类
（1）计算待分类数据点与训练集所有样本点距离最近的k个样本；
（2）统计这k个样本的类别数量；
（3）根据多数表决方案，取数量最多的那一类作为待测样本的类别。

要点：
（1）距离量度可以依据样本不同而改变，通常有：Euclidean distance, Manhattan distance, cosine.(欧式距离，曼哈顿距离，cos距离)
    注意：这里样本间距离可以依据实际情况自己做改动
（2）需要训练集，标明所属类别
（3）距离最近的k个样本，这个k需要依据经验来设定。

测试数据集：http://archive.ics.uci.edu/ml/datasets/Iris    [x1,x2,...,类别],注意类别改为数字

'''


import numpy as np
import scipy.spatial.distance as ssd

def read_data(fn):
    """
        read daataset file
    Args:
        fn: 读取文档的路径，要求文件以’,’分隔
    Returns:
        返回读取的数据集，第一维是数据，第二维是类别。
    """
    # initialize
    charac=[]; label=[];

    with open(fn) as f:
        #raw_data = np.loadtxt(f,delimiter=',',dtype="float",usecols=(0,1,2,3))
        raw_data = np.loadtxt(f,delimiter=',',dtype="float",usecols=None)
    # obtain input characteristics and labels
    for row in raw_data:
        charac.append(row[:-1])
        label.append(int(row[-1]))
        
    return np.array(charac),np.array(label)
    

def knn(k,dtrain,dtest,dtr_label):
    """
        k-nearest neighbors algorithms
    Args:
        k: 标识寻找最近的k个样本
        dtrain: 表示训练集
        dtest: 表示测试集
        dtr_label: 保存所有类别种类，访问时按照不同下标区分
    Returns:
        返回测试集每一个样例的测试结果
    """
    
    pred_label=[]
    
    # 对每一个测试集，找到其与所有训练集的距离
    for di in dtest:
        distances=[]
        for ij,dj in enumerate(dtrain): # enumerate表示遍历数组
            distances.append((ssd.euclidean(di,dj), ij));   # ij表示下标,dj表示value
        
        # sort the distance and get the top-k instances.
        k_nn = sorted(distances)[:k]
        
        # classify according to the maxmium label
        dlabel=[]
        for dis,idtr in k_nn:   # k_nn形式：[打分，下标]
            dlabel.append(dtr_label[idtr])  # 保存预测的类别
        #print dlabel

        pred_label.append(np.argmax(np.bincount(dlabel)))
        # print pred_label
        
    return pred_label


def evaluate(result):
    """
        评估knn聚类的结果
    Args:
        result: 保存预测结果，如果评估正确，保存1，否则保存为0
    Returns:
        长度为2的数组，分别保存评估正确和不正确的样例个数
    """
    eval_result = np.zeros(2,int)
    # 对每一个结果进行评估
    for x in result:
        if x==0:
            eval_result[0]+=1
        else:
            eval_result[1]+=1
    
    return eval_result


def knn_test():
    dtrain,dtr_label=read_data('/Users/a1/Desktop/ali/algorithm_data/knn/knn_train.txt')
    dtest,dte_label=read_data('/Users/a1/Desktop/ali/algorithm_data/knn/knn_test.txt')
    K=[1,2,3,4]
    for k in K:
        pred_label=knn(k,dtrain,dtest,dtr_label)
        eval_result=evaluate(pred_label-dte_label)
        # check the answer
        print k," --> ",eval_result[0],",",eval_result[1]

knn_test()
    
