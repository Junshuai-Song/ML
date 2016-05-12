# coding=utf-8
'''
Created on 2016年5月11日
@author: a1

算法名称：svm支持向量机：SMO序列最小优化算法

工作原理：
    每次循环中选择两个alpha进行优化处理。一旦找到合适的alpha，那么就增大其中一个同时减小另一个。

简化SMP函数的伪代码：
    创建一个alpha向量，初始化为0
    当迭代次数小于最大迭代次数时（外循环）：
        对数据集中的每个数据向量（内循环）：
            如果该数据向量可以被优化：
                随机选择另外一个数据向量
                同时优化这两个向量
                如果两个向量都不能被优化，退出内循环
        如果所有向量都没被优化，增加迭代数目，继续下一次循环

完整版的SMP函数伪代码：
    其与简化版的唯一区别：在于选择alpha的方式在选择第一个alpha值后，算法会通过最大化步长的方式来获得第二个alpha值；
    （以前是选择j之后计算错误率，需要遍历一遍所有样本集）
    
对于线性不可分的情况，我们采用最流行的 径向基核函数 来预处理数据


测试数据集：

参考：
（1）http://blog.csdn.net/alvine008/article/details/9097105
（2）http://blog.csdn.net/alvine008/article/details/9097111

优势：
（1）knn算法需要保存训练样本；这里不需要保存全部训练样本，只需要支持向量的样本即可。大大减少了样本数量;

缺点：
（1）貌似只能做二分类 ？

'''
from math import *
from numpy import *
from time import sleep

def loadDataSet(fileName):
    """
        获取数据集[x1,x2,...,类别]
    """
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    """
        通过一个alpha i，来随机获取另外一个alpha
        在改进方法中，这里不是随机获取的
    Args:
        i:第一个alpha的下标
        m:所有alpha的数目
    Returns:
        返回随机的另一个alpha下标
    """
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j;

def clipAlpha(aj,H,L):
    """
        用于调整大于H或小于L的alpha值
    """
    if aj > H:
        aj=H
    if L > aj:
        aj = L
    return aj 

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
        简化的SMO算法
    Args:
        dataMatIn:数据集
        classLabels:类别标签向量
        C:常数C
        toler:容错率
        maxIter:最大迭代次数
    """
    
    # 获取数据矩阵
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0;
    m,n = shape(dataMatrix) # m*n的矩阵
    alphas = mat(zeros((m,1)))  # m*1列的alpha矩阵列
    
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0   #标记alpha是否在当前轮循环中进行了优化
        for i in range(m):  # 对数据集中每个数据向量
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i,:].T))    # 我们预测的类别
            Ei = fXi - float(labelMat[i])   # 预测误差：如果过大就可以对对应的alpha值进行优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
            ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): #其中对正负间隔都做了判断，如果过大 
                # 如果当前数据向量可以更改,下面随机选取另一个数据向量
                j = selectJrand(i,m)    # 随机选取
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])   #计算其预测误差
                
                # 复制i,j两个alpha的值
                alphaIold = alphas[i].copy()    # 必须加上copy，默认是引用
                alphaJold = alphas[j].copy()
                # 计算L,H，后面会将alpha调整至0-C之间
                if (labelMat[i] != labelMat[j]):    # 实际不同类
                    L = max(0, alphas[j] - alphas[i])   
                    H = min(C, C+alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print "L==H"
                    continue
                
                # eta表示alpha[j]的最优修改量，当其为0时计算较为麻烦，这里简化了处理过程（实际中一般不常发生）
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    dataMatrix[i,:] * dataMatrix[i,:].T - \
                    dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print "eta>=0"
                    continue
                
                # 利用eta/L/H对alphas[j] 进行修正； 之后alphas[i]需要进行大小相同，方向相反的修正
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta 
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += labelMat[j] * labelMat[i]*(alphaJold - alphas[j])
                
                # 这部分没有很理解是做什么的，最后将不断更新计算的b返回了
                b1 = b- Ei - labelMat[i] * (alphas[i]-alphaIold) * \
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold) * \
                    dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold) * \
                    dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold) * \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                
                # 为当前两个alpha设置一个常数项b
                if(0 < alphas[i]) and (C>alphas[i]): b = b1;
                elif (0 < alphas[j]) and(C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                
                alphaPairsChanged += 1
                print "iter: %d  i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            
        # 如果
        if (alphaPairsChanged == 0): iter +=1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas   



# 以下是完整的改进方法
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0  # 以上数据含义同简单版
        self.eCache = mat(zeros((self.m,2)))  #全局缓存，用于保存误差值:第一列标示是否有效；第二列是有效E值
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k):
    """
        对于给定的alpha值，可以计算出对应E值并返回
    """
#     fXk= float(multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
#     Ek = fXk - float(oS.labelMat[k])
    fXk= float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    """
        内循环中的启发式方法，用于选择另外一个j
    """
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  # 返回非0E值所对应的alphas
    if (len(validEcacheList)) > 1 :
        for k in validEcacheList:   # 使用循环，寻找改变最大的那个值
            if k==i: continue
            Ek = calcEk(oS,k)   #误差
            deltaE = abs(Ei-Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej=Ek
        return maxK,Ej
    else:   # 如果是第一次循环，那么就随机选取一个
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j,Ej

def updateEk(oS, k):
    """
        计算误差值，并将其存入缓存中
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
            
def innerL(i, oS):
    """
        选择j，即另一个alpha
    """
    Ei = calcEk(oS,i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i]<oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i]>0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        #注释的为不使用核函数时的代码
#         eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - \
#             oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        
        if eta>=0 : print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        
        # 注意这里的\后面不能有任何字符或者空格
        # 注释的为不使用核函数的代码
#         b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
#             oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
#             (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#         
#         b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)* \
#             oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j] * \
#             (oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
            oS.K[i,i] - oS.labelMat[j]*\
            (oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)* \
            oS.K[i,j] - oS.labelMat[j] * \
            (oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0 

def smoP(dataMatIn, classLabel, C, toler, maxIter, kTup=('lin',0)):
    """
        完整的SMP函数
    """
    oS = optStruct(mat(dataMatIn), mat(classLabel).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    # 当迭代到大最大值，或者遍历整个集合都未对任意alpha对进行修改时，退出循环
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):   # 遍历任意可能的alpha
                alphaPairsChanged += innerL(i,oS)   # 调用innerL来选择第2个alpha
            print "fullSet, iter: %d  i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            #iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:    # 遍历所有非边界值
                alphaPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d  i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)
        iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas




# 至此，我们得到了大量的alpha值，那么之后我们如何使用这些alpha值来得到超平面，进而进行分类呢？

def calcWs(alphas, dataArr, classLabels):
    """
        用于计算得到的大量alpha值来计算超平面
        其中非0alphas对应的即为支持向量
    """
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m): # 遍历了所有向量，但是起作用的只有支持向量
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def test():
    """
        没有核函数的情况，参数为'lin'
    """
    dataArr,labelArr = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40, ('lin', 1))
    ws = calcWs(alphas, dataArr, labelArr)
    print ws

    # 对数据进行分类
    dataMat = mat(dataArr)
    if(dataMat[0]*mat(ws) + b > 0): # 对于1好元素，如果取值大于0，则为第一类；
        print '1'
    else:   #否则为第2类
        print '-1'
    
    # 之后与labelArr对应下标分类值做对比，即可观察分类正确性

#test()

'''
    现实中，有事不是直接线性可分的情况；
    这种时候需要我们 核函数 将数据转换成分类器易理解的形式
    
下面我们提出：径向基核函数
    一种最流行的核函数
    科学家希望将这种方式称为：将数据从一个特征空间转到另一个特征空间
    通常情况下是低维 --> 高维
    把内积运算替换成核函数
'''

def kernelTrans(X, A, kTup):
    """
        径向基核函数：采用向量作为自变量，能够基于向量距离运算输出一个标量。
        注意核函数需要相应修改部分innerL()/calcEk()函数中部分代码
    """
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X*A.T
    elif kTup[0]=='rbf':    # 径向基核函数：在for循环中对矩阵中每个元素计算高斯函数的值
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K / (-1*kTup[1]**2))
    else:
        raise NameError('Houston We have a Problem  - - That Kernel is not recognized')
    return K



def test_kernel(k1 = 1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    # 这里的几个参数都可以进行调整
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A >0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))    #先将数据利用核函数转换
        predict = kernelEval.T * multiply(labelSV, alphas[svInd] + b)
        if sign(predict)!=sign(labelArr[i]): errorCount+=1
    print "the training error rate is: %f" % (float(errorCount)/m)
    
    # 换另外一个测试集
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf',k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)
    
    

    # 之后与labelArr对应下标分类值做对比，即可观察分类正确性

test_kernel(0.09)
# 在实际中，我们可以通过调整k1值得到不同的结果。需要调参的过程。
# 参数还包括smop函数的参数:常数C/toler:容错率/maxIter:最大迭代次数，在文件361行

