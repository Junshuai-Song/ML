# coding=utf-8
'''
Created on 2016年5月20日
@author: a1

算法名称：
（1）

要点：
（1）

测试数据集：

'''

from numpy import *
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


ans = datelist((2015, 9, 1), (2015, 10, 31))
print shape(ans)[0]
for i in range(shape(ans)[0]):
    print ans[i]
    
    