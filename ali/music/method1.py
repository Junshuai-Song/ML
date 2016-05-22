# coding=utf-8
'''
Created on 2016年5月16日
@author: a1

特征：

只下载或收藏，但是没有听过的歌曲，我们当做垃圾数据，直接删除。（对于只有收藏、下载，但是没有一次播放的歌，直接删除）

每首歌每个用户的平均播放时间
每首歌的播放/播放次数，前1-7天
每首歌的下载/播放次数
每首歌的下载/播放人数
每首歌的收藏/播放次数
每首歌的收藏/播放人数


同一艺人某一首歌突然大卖，其老歌播放量会有回升。

歌曲总数

预测之后60天每位艺人所有歌曲的播放总量。

最后，不同语言的歌曲分开建模。（考虑听中文、英文歌曲的人的习惯可能不同）
男女可能也可以分开


总体思路，为每个歌手建模，分别做多参数变量的回归。注意[x1,x2,...,xn]都需要是整数，比如日期，需要换算成整数
（1）Number=41时，有5个缺失值，直接补上；     
    2b7fedeea967bec 20150429 13
    2b7fedeea967bec 20150519 4
    2b7fedeea967bec 20150522 5
    2b7fedeea967bec 20150725 5
    2b7fedeea967bec 20150731 3
（2）找的特征：
    周几；
    当天是否放假
    明天是否上班
    明天是否放假
    
（3）大波峰全部去掉，因为准确地预测到他在哪一天会突然暴增、暴增多少都不好预测。还是删除掉最好。
    找到每一组top5的点，直接删除
    
'''

from algorithms import *   # 如果要明确引入包名下的某个模块，需要再加.，例如：from algorithms.regTrees import *
from numpy import *

if __name__ == "__main__":
    # 从train/ans1-50.txt为50个音乐人的数据集，需要我们分别建模，然后输入之后60天的测试数据
    ans = 0
    for i in range(51):
        if i==0 : continue
        # 使用某个音乐人的数据集，训练模型树
        trainMat = mat(regTrees.loadDataSet('/Users/a1/Downloads/train/ans'+str(i)+'.txt'))
        myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1,80))  # 这里两个误差的控制！需要再调研一下
        
        # 使用训练好的模型树，对测试集进行测试，并输出结果
        testMat = mat(regTrees.loadDataSet('/Users/a1/Downloads/test/test.txt'))
        yHat = regTrees.createForeCast(myTree, testMat[:,0:4], regTrees.modelTreeEval)
        for j in range(shape(yHat)[0]):
            print yHat[j,0]
        
    # 使用训练好的模型树，测试对训练集的拟合程度
        #t = corrcoef(yHat, testMat[:,1],rowvar=0)[0,1]
        #ans += abs(t)
    #print ans   # 输出总体50个样本的拟合效果和
    

        
