# 遍历文件寻找单条目的特征
# coding:utf-8
import time
import numpy as np
import sys
import datetime
import pandas as pd
import os
import random
# 聚类库
from scipy.cluster.vq import *
import readData



#将train训练集提取出5000条特征，（按类的数量来分配），即，38200个train文件随机选5000个txt，txt中随机选300行数据
#300行来聚类，5000个来形成特征
# 5000条特征从7*24维和7*26维里找出。剩下的维度投影到二维。
#2种维度组合分别聚类，聚类中心个数需提前给定。
#输出看聚类不要中心化。看能不能找出特征
#7*24维记得归一化，

# 每个文件的特征行
featherNum = 300
# 单行最少点数,此处用最小长度表示，单行字符串长度小于这个参数时不被记录为特征
minLineLen = 30
# 聚类迭代最大次数
MaxKmeans = 100
# 7*24特征个数
feather7x24 = 4
# 7*26特征个数
feather7x26 = 3


#初始矩阵
date2position = {}
datestr2dateint = {}
str2int = {}


for i in range(24):
  str2int[str(i).zfill(2)] = i

for i in range(182):#天数变成第几周和周几
  date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
  date_int = int(date.__str__().replace("-", ""))
  date2position[date_int] = [i%7, i//7]#余数，整数
  datestr2dateint[str(date_int)] = date_int

#聚类
def clusterK(resnp7x24, resnp7x26):
  k = min(len(resnp7x24), 3)
  centroids7x24, variance = kmeans(resnp7x24, k)  # 得到聚类结果
  centroids7x24 = centroids7x24
  result7x24 = centroids7x24.astype(np.int)
  print('result7x24')

  k = min(len(resnp7x26), 3)
  centroids7x26, variance = kmeans(resnp7x26, k)  # 得到聚类结果
  centroids7x26 = centroids7x26
  result7x26 = centroids7x26.astype(np.int)
  print('result7x26')

  # k = min(len(resnp24), 3)
  # centroids24, variance = kmeans(resnp24, k)    #得到聚类结果
  # centroids24=24*centroids24
  # result24 = centroids24.astype(np.int)
  # print('result24')
  # print(result24[0], "\t", result24[1], "\t", result24[2], "\t")
  # print('result24 length'+len(resnp24))

  # k = min(len(resnp26), 3)
  # centroids26, variance = kmeans(resnp26, k)    #得到聚类结果
  # centroids26=26*centroids26
  # result26 = centroids26.astype(np.int)
  # print('result26')
  # print(result26[0], "\t", result26[1], "\t", result26[2], "\t")
  # print('result26 length'+len(resnp26))

  return result7x24, result7x26

# 读取Train数据，聚类,结果按类别存储
def getTrainCluster():
  #把聚类结果按照分类来存储
  lists=readData.readTrainData2Mem()#list[class][file][content]
  for classnum in range(0,9):
    cluresults7x24 = {}
    cluresults7x26 = {}
    for filenum in range(0,lists.shape(1)):
      for contentnum in range(0,lists.shape(2)):
        visitdata = lists[classnum][filenum][contentnum]
        rec7 = []
        rec24 = []
        rec26 = []
        rec7x24 = np.zeros((7, 24))
        rec7x26 = np.zeros((7, 26))
        for item in visitdata.split(','):
          temp = []
          temp.append([item[0:8], item[9:].split("|")])#日期，几点
          for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]#x余数周几，y整数第几周
            rec7x26[x][y] += 1
            rec7.append([x])
            rec26.append([y])
            for visit in visit_lst:#几点
              rec7x24[x][str2int[visit]] += 1
              rec24.append([str2int[visit]])

        temp1 = []
        for i in range(7):
          for j in range(24):
            temp1.append([i, j, rec7x24[i][j]])
        temp2 = []
        for i in range(7):
          for j in range(26):
            temp2.append([i, j, rec7x26[i][j]])

          #均一化？？
        resnp7x24= np.array(temp1)
        resnp7x26 = np.array(temp2)
        # resnp7 = np.array(temp, float)/7
        #resnp24 = np.array(rec24, float)/24
        #resnp26 = np.array(rec26, float)/26

        result7x24={}
        result7x26={}
        result7x24, result7x26 = clusterK(resnp7x24, resnp7x26)
        cluresults7x24.append(result7x24)
        cluresults7x26.append(result7x26)

    np.save("../data/cluster/train_7x24/class_" + classnum + ".npy", cluresults7x24)
    np.save("../data/cluster/train_7x26/class_" + classnum + ".npy", cluresults7x26)
    sys.stdout.write('\r>> train cluster classnum: %d' % classnum)
    sys.stdout.flush()


# 读取Test数据，聚类，结果按文件存储
def getTestCluster():
  lists=readData.readTrainData2Mem()#2维list[file][content]
  cluresults7x24 = {}
  cluresults7x26 = {}
  for filenum in range(0,lists.shape(0)):
    for contentnum in range(0,lists.shape(1)):
      visitdata = lists[filenum][contentnum]
      rec7 = []
      rec24 = []
      rec26 = []
      rec7x24 = np.zeros((7, 24))
      rec7x26 = np.zeros((7, 26))
      for item in visitdata.split(','):
        temp = []
        temp.append([item[0:8], item[9:].split("|")])#日期，几点
        for date, visit_lst in temp:
          x, y = date2position[datestr2dateint[date]]#x余数周几，y整数第几周
          rec7x26[x][y] += 1
          rec7.append([x])
          rec26.append([y])
          for visit in visit_lst:#几点
            rec7x24[x][str2int[visit]] += 1
            rec24.append([str2int[visit]])

      temp1 = []
      for i in range(7):
        for j in range(24):
          temp1.append([i, j, rec7x24[i][j]])
      temp2 = []
      for i in range(7):
        for j in range(26):
          temp2.append([i, j, rec7x26[i][j]])

          #均一化？？
      resnp7x24= np.array(temp1)
      resnp7x26 = np.array(temp2)
      # resnp7 = np.array(temp, float)/7
      #resnp24 = np.array(rec24, float)/24
      #resnp26 = np.array(rec26, float)/26

      result7x24={}
      result7x26={}
      result7x24, result7x26 = clusterK(resnp7x24, resnp7x26)
      cluresults7x24.append(result7x24)
      cluresults7x26.append(result7x26)

    #此时用filenum好不好？还是去test_list找对应num的文件名？
    np.save("../data/cluster/test_7x24/" + filenum + ".npy", cluresults7x24)
    np.save("../data/cluster/test_7x26/" + filenum + ".npy", cluresults7x26)
    sys.stdout.write('\r>> test cluster filenum: %d' % filenum)
    sys.stdout.flush()



# 主程序
if __name__ == '__main__':
  if not os.path.exists("../data/cluster/train_7x24/"):
    os.makedirs("../data/cluster/train_7x24/")
  if not os.path.exists("../data/cluster/train_7x26/"):
    os.makedirs("../data/cluster/train_7x26/")
  if not os.path.exists("../data/cluster/test_7x24/"):
    os.makedirs("../data/cluster/test_7x24/")
  if not os.path.exists("../data/cluster/test_7x26/"):
    os.makedirs("../data/cluster/test_7x26/")

  num=500
  readData.readTrainData2File(num)
  getTrainCluster()

  readData.readTestData2File(num)
  getTestCluster()