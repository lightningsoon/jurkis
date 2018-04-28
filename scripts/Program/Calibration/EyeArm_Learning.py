# -*- coding=utf-8 -*-
# import sklearn
import csv

from calibrate import ArmEye_collectingData
# a=ArmEye_collectingData()
#
# print(len(list(a.para)))
class Pinocchio(object):
    def __init__(self):
        self.__f_para = open('data.csv', 'r')  # 数据
        self.__f_coor = open('label.csv', 'r')  # 标签
        self.__writer_para = csv.writer(self.__f_para)
        self.__writer_coor = csv.writer(self.__f_coor)
    def train(self):
        pass
    def test(self):
        pass
    def dumpModel(self):
        pass
    def restoreModel(self):
        pass

    def predict(self,coor):
        return 0
        pass