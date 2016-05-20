# coding=utf-8
'''
Created on 2016年5月11日
@author: a1

算法名称：Logistic回归

一般过程
（1）收集数据：采用任意方法
（2）准备数据：由于需要进行距离计算，因此要求数据类型为数值型
（3）分析数据：任意方法对数据进行分析
（4）训练算法：为了找到最佳的分类回归函数
（5）测试算法：一旦训练完成，分类将会很快
（6）使用算法：判断样例属于哪个类别。注意在这之后，我们就可以在输出的类别基础上进行一些其他的分析工作

梯度上升法步骤：
    每个回归系数初始化为1
    重复R次：
        计算整个数据集的梯度
        适用alpha * gradient更新回归系数的向量
        返回回归系数

随机梯度上升法步骤：
    所有回归系数初始化为1
    对数据集中每个样本：
        计算该样本的梯度
        使用alpha * gradient更新回归系数的向量

要点：
（1）适用数值型和标称型数据
（2）正常的随机梯度算法复杂度太高，在实际运用中我们使用随机梯度上升算法，即一次仅用一个样本点来更新回归函数
（3）改进的随机梯度上升算法会导致每次计算结果不太一样，这是因为其中随机选取样本的原因

测试数据集：

优点：易于理解和实现，计算代价不高
缺点：容易欠拟合，分类精度可能不高。
'''
from math import *
from numpy import *
from scipy.spatial.distance import *
import matplotlib.pyplot as plt

def loadDataSet():
    """
        获取数据集[x1,x2,...,类别]
    """
    dataMat = []; labelMat = []
    
    with open('testSet.txt') as f:
        #raw_data = np.loadtxt(f,delimiter=',',dtype="float",usecols=(0,1,2,3))
        raw_data = loadtxt(f,delimiter='\t',dtype="float",usecols=None)
    # obtain input characteristics and labels
    for row in raw_data:
        dataMat.append([1.0,row[0],row[1]])
        labelMat.append(int(row[-1]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
        梯度上升算法，求解回归系数
    Args:
        dataMatIn:训练样本，一行一个样本，多个特征
        classLabels:样本所属类别
    Returns:
        
    """
    dataMatrix = mat(dataMatIn) # 获取输入数据，将其转换为NumPy矩阵
    labelMat = mat(classLabels).transpose() # 将类别向量转置并转换为NumPy矩阵
    m,n = shape(dataMatrix) # m*n的矩阵
    
    alpha = 0.001   # 迭代参数之一alpha，向目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n,1))
    
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   # 矩阵相乘[m,3]*[3,1]=[m,1]
        error = (labelMat - h)  # 计算值是类别与预测类别的差值
        # 按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose()*error 
        
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    """
        随机梯度上升算法获取回归系数
    """
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
        改进的随机梯度上升算法获取回归系数
        多轮迭代，每轮迭代随机选取样本
        每轮迭代alpha调整，一直减小，但不会到0
    """
    m,n = shape(dataMatrix)
    weights = ones(n)
    
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])   # 每次从队列中选取一个值，之后删除
    return weights


def plotBestFit(wei):
    """
        画出数据集和Logistic回归最佳拟合直线
    """
    
    weights = wei.getA()
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1=[]; ycord1=[]
    xcord2=[]; ycord2=[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    

def logistic_test():
    dataArr,labelMat = loadDataSet()
    
    #weights = gradAscent(dataArr, labelMat)
    #weights = stocGradAscent0(array(dataArr), labelMat)
    weights = stocGradAscent1(array(dataArr), labelMat)
    print weights
    
    #plotBestFit(weights)
    plotBestFit(mat(weights).transpose())
    

logistic_test()


