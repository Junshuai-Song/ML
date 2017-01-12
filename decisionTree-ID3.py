# coding=utf-8
'''
Created on 2016年5月10日
@author: a1

算法名称：decisionTree决策树-ID3算法，属于监督分类算法
    ID3是一种典型的决策树算法，C4.5，CART都是在其基础上发展而来。
    决策树的叶子节点表示类标号，非叶子节点作为属性测试条件。
    从树的根节点开始，将测试条件用于检验记录，根据测试结果选择恰当的分支，直到达到叶子节点，叶子节点的类标号即为该记录的类别
    
    ID3采用信息增益作为分裂属性的度量，最佳分裂等价于求解最大的信息增益

流程：
（1）如果节点的所有类标号相同，停止分裂；
（2）如果没有feature可供分裂，根据多数表决确定该节点的类标号，并停止分裂；
（3）选择最佳分裂的feature，根据选择的feature的值逐一进行分裂；递归地构造决策树。

ID3算法步骤：
（1）对当前例子集合，计算各属性的信息增益；（期间需要计算信息熵）
（2）选择信息增益最大的属性A
（3）把A处取值相同的例子归于同一子集，A取几个值就得几个子集；
（4）对即含正例又含反例的子集，递归调用建树算法；
（5）若子集仅含正例或反例，对应分支标上P或N，返回调用处。

要点：
（1）变量的不确定性越大，熵也就越大。所以信息熵可以说是系统有序化程度的一个度量

测试数据集：自己构建    [x1,x2,...,类别]

参考内容：
（1）决策树概念理解：http://blog.csdn.net/alvine008/article/details/37760639
（2）信息熵解释：http://baike.baidu.com/link?url=_50UcLmVpIh35u9HTo4oH707Y8y8uNwC-2UfbRdgH_Opwz9bxP9rQXYis6iQFL849enPWP1inJE5J7fR6g32Ea#3_3


'''

from math import log
import operator
import matplotlib.pyplot as plt

def calcShannonEnt(dataSet):
    """
        计算信息熵
    Args:
        dataSet:输入的数据集
    Returns:
        返回计算的当前集合信息熵
    """
    numEntries=len(dataSet)
    
    # 找到所有的label类别标签，记录出现次数
    labelCounts={}
    for entry in dataSet:
        currentLabel=entry[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0  #第一次加入到labelCounts中，先赋值为0
        labelCounts[currentLabel]+=1 

    # 记录各个类别的样例个数
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)    # log函数在0-1之间为负数
    
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    """
        按照给定的特征划分数据集：删除第axis维度的特征，特征值为value
    Args:
        dataSet:数据集
        axis:第几维度的特征下标,需要删除该维度的特征
        value:给定的特征
    Returns:
        返回数据集中axis维度特征=value的所有数据集，去掉该维特征后的数据
    """
    retDataSet=[]
    for entry in dataSet:
        if entry[axis]==value:
            reduced_entry=entry[:axis]
            reduced_entry.extend(entry[axis+1:])    # reduced_entry获取entry[axis-end]数据
            retDataSet.append(reduced_entry)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
        计算最大信息增益的属性下标
    """
    numFeatures=len(dataSet[0])-1   #去掉最后类别维度
    baseEntropy = calcShannonEnt(dataSet)   #计算当前集合信息熵
    bestInfoGain = 0.0; bestFeature = -1;   #初始化最优划分属性
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #获取第i维度的一列所有数据特征
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:        #按照某一列的属性分为各个数据集，计算此种方式下的信息增益
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy #按第i维属性划分方式下的信息增益
        
        if( infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  #返回当前数据集最优的属性划分方式

def majorityCnt(classList):
    """
        找出出现次数最多的分类名称
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
        创建决策树
    Args:
        dataSet:当前数据集
        labels:当前数据集包含所有特征
    """
    classList = [example[-1] for example in dataSet]    #classList保存所有出现的类别
    if classList.count(classList[0]) == len(classList): # 当前样例所有类所属类别一样，则停止递归
        return classList[0] 
    if (len(dataSet[0])==1):    # 当前还存在多种类别，但是没有更多的属性来划分了
        return majorityCnt(classList)   # 所以拿出现最多频率的类别当做该部分样例的类别
    
    # 继续递归
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]    # 使用计算出的下标，来获取对应特征label
    myTree = {bestFeatLabel:{}} # 初始化该子树
    del(labels[bestFeat])   # 删除当前最优的特征
    
    # 获取目前选择的最优特征的所有属性
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)    # 建立一个set object
    for value in uniqueVals:
        subLabels = labels[:]   #已经删除了最优的特征    
        # !!! 这里使用一个新变量接收label是因为python的引用传递机制，如果这里直接使用label，那么各个递归节点之间会互相产生影响
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def decisionTree_test():
    myDat,labels = createDataSet()
    myTree = createTree(myDat, labels)
    return myTree

#ans = decisionTree_test()
#print ans

def classify_test(inputTree, featLabels, testVec):
    """
        使用以上建立的决策树进行分类
    """
    firstStr = inputTree.keys()[0]  # 获得'no surfacing'
    secondDict = inputTree[firstStr]    # 子树
    featIndex = featLabels.index(firstStr)  # 'no surfacing'下标为0
    for key in secondDict.keys():   # 0 1
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':    # 包含子树
                classLabel = classify_test(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

    
myDat,labels = createDataSet()
inputTree = createTree(myDat, labels)   #注意这里传入的是引用，labels被改变了
myDat,labels = createDataSet()
testVec = [1,0] 
ans = classify_test(inputTree,labels,testVec)
print ans


