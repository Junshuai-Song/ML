# coding=utf-8
'''
Created on 2016年5月17日
@author: a1

算法名称：回归树 & 模型树
    相当于决策树的变种，可以预测数值

要点：
（1）一样需要防止过拟合、欠拟合的情况
（2）原程序有两个小bug，这里做了修复。

测试数据集：

'''

from numpy import *
from numpy import mat

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
#     print nonzero(dataSet[:,feature] > value)[0]
#     mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0] #g 当某
#     mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
#     print mat0
#     print ("|")
#     print mat1
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    number = 0
    for featIndex in range(n-1):
        l = dataSet.tolist()
        column = [col[featIndex] for col in l]
#         print set(column)
        for splitVal in set(column):
#             print "number = ", number
            number += 1
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]  # 表示inDat有几列
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat



import datetime

def datelist(start, end):
    """
        生成时间列表，形,如:(2012,12,12),(2013,12,12)
    Args:
        start:起始时间
        end:终止时间
    Returns:
        返回起始时间与终止时间的一个时间列表
    """
    start_date = datetime.date(*start)
    end_date = datetime.date(*end)

    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append(int("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day)))
        curr_date += datetime.timedelta(1)
    result.append(int("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day)))
    return result

if __name__ == "__main__":
    #ans = datelist((2015, 9, 1), (2015, 10, 31))
    #print shape(ans)[0]
#     for i in range(shape(ans)[0]):
#         print ans[i]

    # 从train/ans1-50.txt为50个音乐人的数据集，需要我们分别建模，然后输入之后60天的测试数据
    ans = 0
    for i in range(51):
        #print i
        if i==0 : continue
        trainMat = mat(loadDataSet('/Users/a1/Downloads/train/ans'+str(i)+'.txt'))
        myTree = createTree(trainMat, modelLeaf, modelErr, (1,80))  # 这里两个误差的控制！需要再调研一下
        #print myTree
        
        # 测试，注意需要更改
        testMat = mat(loadDataSet('/Users/a1/Downloads/test/test.txt'))
        yHat = createForeCast(myTree, testMat[:,0:4], modelTreeEval)
        #print shape(yHat)
        for j in range(shape(yHat)[0]):
            print yHat[j,0]
        #print yHat
        #t = corrcoef(yHat, testMat[:,1],rowvar=0)[0,1]
        #print t
        #ans += abs(t)
    #print ans   # 输出总体50个样本的拟合效果和
# 现在拟合的效果都很差！！！ 
# 现在基本差不多了，还需要再多提几个特征！  
# 之后如果还很差的话，我们需要改用随机森林构造的回归树！！！


        
        
        
        
        
        
