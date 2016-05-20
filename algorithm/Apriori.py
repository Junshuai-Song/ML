# coding=utf-8
'''
Created on 2016年5月13日
@author: a1

算法名称：Apriori进行关联分析
    寻找不同的物品组合是一件十分耗时的事情，计算代价太高；
    蛮力搜索不能解决这个问题，需要更加智能的方法在合理的时间找到频繁项集合。

步骤：
    收集数据
    准备数据
    分析数据
    训练数据：使用Apriori算法来找到频繁项集
    测试算法：不需要测试过程
    使用算法：用于发现频繁项集以及物品之间的关联规则
    
Apriori原理：
    如果某个项集是频繁的，那么它的所有子集也是频繁的。
    反之亦可：如果某个项集是非频繁集，那么它的所有超集也是非频繁的。
    
伪代码：
    当集合中项的个数大于0时，循环
        构建一个k个项组成的候选集的列表
        检查数据以确认每个项集都是频繁的
        保留频繁项集并构建k+1项组成的候选项集的列表
    

要点：
（1）

测试数据集：

优点：易于实现
缺点：在大数据集上可能较慢
适用于数值型或标称型数据

'''

def loadDataSet():
    """
        创建了一个用于测试的简单数据集，可以认为是几次购买情况
    """
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    """
       创建第一个候选项集的列表C1
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:    # 第一个候选项集C1中元素不重复
                C1.append([item])   # 这里添加的不是单个物品项，而是一个只含该物品的列表
    C1.sort()
    return map(frozenset, C1)   # 指冰冻集合，即不可更改的集合

def scanD(D, Ck, minSupport):
    """
        用来对D中的候选集进行检测，支持度是否大于minSupport
    Args:
        D:包含候选集集合的列表
        Ck:数据集
        minSupport:需要保留的最小支持度
    """
    ssCnt = {}  # 先构建一个空字典
    for tid in D:   # 遍历候选集
        for can in Ck:  # 对每一个候选集，需要遍历一遍数据集，看其出现情况，并统计在字典中
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] +=1
                
    # 字典ssCnt保存了各个候选集出现的次数，之后用来计算每个候选集的支持度
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:   #对每一个在数据集中出现过的候选集，计算支持度
        support = ssCnt[key]/numItems   
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support  # 返回最频繁项集的支持度
    return retList, supportData

def aprioriGen(Lk, k):
    """
        依据Lk来创建候选集Ck
    Args:
        Lk:频繁项集列表
        k:项集元素个数
    """
    # 例如，输入{0},{1},会输出{0,1}
    retList = []
    lenLK = len(Lk)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    """
        
    """
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1,supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2])>0):
        Ck = aprioriGen(L[k-2], k)  # 利用L来创建新的候选集
        Lk, supK = scanD(D, Ck, minSupport) # 对候选集进行筛选
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData    



def test():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet, 0.5)
    print L
    print suppData

test()












