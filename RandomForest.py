# coding=utf-8
'''
Created on 2016年5月15日
@author: a1

算法名称：随机森林,这里我们先使用sklearn包含的随机森林库，之后尝试自己实现
    随机森林就是通过集成的学习思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习
    的一大分支：集成学习方法。
    森林：多颗决策树
    随机：多颗决策树参与训练样本、特征随机选择。

每棵树生成规则：
（1）每棵树，随机且有放回地从训练集中抽取N个训练样本作为训练集；
（2）每个样本的特征维度为M，我们随机选取m(m<<M)个特征进行决策树建立；
（3）每棵树就尽最大程度地生长，且没有剪枝过程。

要点：
（1）森林中任意两颗数的相关性：相关性越大，错误率越大；
（2）森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。
减小特征选择个数m，树之间的相关性和每棵树的分类能力都会相应降低；
增大m，两者会同步增大。
所以关键问题就是如何选择最优的m，这也是随机森林唯一的一个参数。

参考资料：http://www.cnblogs.com/maybe2030/p/4585705.html

'''

from math import log
import operator
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression problem
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score
def createDataSet():
    dataSet=[[1,1,1],
             [1,1,2],
             [1,0,1],
             [0,1,1],
             [0,1,1]]
    labels=[1,1,0,0,0]
    return dataSet,labels
X,y = createDataSet()
model.fit(X, y)
#Predict Output
predicted= model.predict([[1,1,1],[2,2,2],[3,3,3.9]])

print predicted

