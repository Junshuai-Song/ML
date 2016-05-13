# coding=utf-8
'''
Created on 2016年5月13日
@author: a1

算法名称：预测数值型变量的线性回归算法（这里暂时不包含非线性回归）
（1）线性回归的基本方法
（2）局部平滑技术，分析如何更好地拟合数据；
（3）“欠拟合”情况下的缩减技术；
（4）融合所有技术，预测鲍鱼年龄和玩具售价


测试数据集：

缺点：对非线性的数据拟合不好

'''

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
        读取数据，分别为数据和最终的目标值
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr, yArr):
    """
        普通最小二乘法，利用NumPy库里的矩阵方法，可以仅使用几行代码来完成所谓的OLS方法
    """
    xMat = mat(xArr); yMat = mat(yArr).T    # 将数据转换为矩阵
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # 当行列式结果为0，表示不可逆，直接计算回归值会出错
        print "This matrix is singular, cannot do inverse"
        return 
    ws = xTx.I * (xMat.T * yMat)
    return ws

def test_normal():
    """
        最小二乘法的测试
    """
    xArr,yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    print ws
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws  #预测结果值,使用ws系数与其相乘等到
    
    # 为了计算拟合效果，我们计算预测数据与原始数据的回归系数
    print corrcoef(yHat.T, yMat)
    # 对于测试数据，回归系数为0.98，可以看出两者效果还是不错的
    
    # 创建了图像 & 画出了原始数据
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    # 画出最佳的拟合直线
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()


#test_normal()

# 下面介绍局部加权线性回归：如利用潜在的一些波动变化模式
def lwlr(testPoint, xArr, yArr, k=1.0):
    """
        局部加权线性回归
        其中提到了一个核的概念，最常用的是高斯核：
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m))) # 创建对角权重矩阵
    for j in range(m):  # 算法遍历数据集，计算每个样本对应的权重：随着样本点与待预测点距离的递增，权重指数衰减
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0*k**2))  #权重值大小以指数级衰减，k控制衰减速度
    
    xTx = xMat.T * (weights * xMat) # 
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse."
        return 
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
        为数据集中每个点调用lwlr，有助于求解k的大小
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)   # 对于给定空间中的任意一点，计算其预测值
    return yHat

def rssError(yArr, yHatArr):#计算真实值与与测试的误差，数值越大，误差越大
    return  ((yArr-yHatArr)**2).sum()   

def test_lwlr():
    xArr,yArr = loadDataSet('ex0.txt')
#     print lwlr(xArr[0],xArr,yArr,1.0)
#     print lwlr(xArr[0],xArr,yArr,0.001) 
    # yHat就是预测结果，可以拿来与真实值进行对比
    yHat = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.05)    # 这里k=1是，效果与最小二乘法一样，属于欠拟合；
    # 0.01时效果较好；
    # 0.003时，拟合的曲线与原数据过于拟合
    print rssError(yArr[0:99], yHat.T)
    
    # 在新的数据集上进行测试
    yHat = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 0.05) 
    print rssError(yArr[100:199], yHat.T)
    
    # 画图
#     xMat = mat(xArr)
#     srtInd = xMat[:,1].argsort(0)   #首先对xArr排序
#     xSort = xMat[srtInd][:,0,:] 
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(xSort[:,1],yHat[srtInd])
#     ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
#     plt.show()
    
    
test_lwlr()

