# coding=utf-8
'''
Created on 2016年5月11日
@author: a1

算法名称：贝叶斯分类，监督分类方法

假设：
（1）每个单词或特征相互独立，即：假设每个单词出现的概率与其相邻词没有关系；
（2）每个特征同等重要。
以上两个假设都存在瑕疵，但是其实际效果还不错。

步骤：
（1）准备数据：从文本中构建词向量
（2）从词向量计算概率
    计算每个类别中的文档数目
    对每篇训练文档：
        对每个类别：
            如果词条出现文档中-->增加该词条的计数值
            增加所有词条的计数值
        对每个类别：
            对每个词条：
                将该词条的数目除以总词条数目得到条件概率
        返回每个类别的条件概率

要点：
（1）

测试数据集：

优点：
（1）在数据集较少时仍然有效；

缺点：
（1）对于输入数据的准备方式较为敏感
（2）适用数据类型：标称型数据
'''

from numpy import *
from scipy.stats.morestats import bayes_mvs

def loadDataSet():
    """
        创建了一些实验样本，classVec表示该词条是否为侮辱性留言
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    """
        获得词汇表：
        利用set去除dataSet中重复元素,得到一个包含全部文档所有词的列表（注意是全部训练文档）
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
        获得词汇表后使用
    Args:
        vocabList:词汇表
        inputSet:测试的某个文档
    Returns:
        返回输入文档对应的向量
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 出现过的word，标记为1
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    """
        setOfWords2Vec的改进，
        统计每个词出现的次数，而不仅仅是标记是否出现
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
    

def trainNB0(trainMatrix, trainCategory):
    """
        贝叶斯分类训练函数
    Args:
        trainMatrix:文档矩阵
        trainCategory:每篇文档类别，构成的向量
    Returns:
        
    """
    numTrainDocs = len(trainMatrix) # 有多少个文档
    numWords = len(trainMatrix[0])  # 特征个数（即所有文档的所有词条，又叫词汇表）
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    
    # 初始化分子分母变量
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    
    for i in range(numTrainDocs):
        # 对于每一篇文档
        if trainCategory[i] == 1:   # 如果需要判断的类别较多，那么这里需要稍加修改
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
        朴素贝叶斯分类函数
    Args:
        vec2Classify:待测试文档向量
        p0Vec:由trainNB0函数得出的三个结果之一
        p1Vec:由trainNB0函数得出的三个结果之一
        pClass1:由trainNB0函数得出的三个结果之一
    """
    p1 = sum(vec2Classify * p1Vec)
    p0 = sum(vec2Classify * p0Vec)
    if p1>p0:
        return 1
    else:
        return 0


def bayes_test():
    listPosts,listClassed = loadDataSet()   # 加载预先处理的多个文档 & 对应类别
    myVocabList = createVocabList(listPosts)    # 获得词汇表
    trainMat = []
    for postinDoc in listPosts: # 使用for循环来获取多个文档对应的矩阵
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat, listClassed)   # 得到训练结果
    
    # 测试文档
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) # 获取文档矩阵
    print testEntry,'classified as: ',classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid','grabage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) # 获取文档矩阵
    print testEntry,'classified as: ',classifyNB(thisDoc, p0V, p1V, pAb)


bayes_test()
