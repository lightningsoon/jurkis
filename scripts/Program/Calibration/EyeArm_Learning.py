# -*- coding=utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


# from calibrate import ArmEye_collectingData
# a=ArmEye_collectingData()
#
# print(len(list(a.para)))
class Pinocchio(object):
    def __init__(self):
        self.__reg_type = 'P'  # 回归器类型
        self.__first_predict=True
        self.__poly = preprocessing.PolynomialFeatures(degree=16)
    def train(self):
        self.readData()
        X,Y=self.X,self.Y
        # Training neural network
        # if self.__reg_type == 'N':
        #     net = rg.Net()
        #     net.train(X, Y, learnrate=0.001, max_iter=10000)

        # Training polynomial regression model
        if self.__reg_type == 'P':
            
            X = self.__poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X, Y)
            self.__dumpModel__(model)
            self.__model=model
            self.test(X, Y)
            return 1

        # Training SVM model
        # if self.__reg_type == 'S':
        #     svm = rg.SVM()
        #     svm.train(X, Y, kernel='linear')
        pass
    def readData(self,dir_x='data.csv',dir_y='label.csv'):
        X = np.loadtxt(dir_x, delimiter=',')
        Y = np.loadtxt(dir_y, delimiter=',')[:,2:]#前两维没有用，检查数据使用
        nonzero_indices=np.where(X[:,-1]!=0)
        # print(nonzero_indices)
        self.X,self.Y=X[nonzero_indices],Y[nonzero_indices]

    def test(self,x,y):
        #x,y:list n*d
        x,y=np.array(x),np.array(y)
        print ("score:", self.__model.score(x, y))
        pass

    def __dumpModel__(self, model):
        if self.__reg_type == 'P':
            joblib.dump(model, './Model/Poly/model.m')
        return 1
        pass

    def __restoreModel__(self):
        try:
            model = joblib.load('./Model/Poly/model.m')
        except:
            print("Import model error!")
            exit()
        return model
        pass

    def predict(self, X_test):
        # X_test：int[3] (3,)
        # Y:int[4] (4,)
        # Using neural network model
        # if self.__reg_type == 'N':
        #     net = rg.Net()
        #     return net.predict(X_test)

        # Using polynomial regression model
        if self.__reg_type == 'P' and self.__first_predict:
            self.__model = self.__restoreModel__()
            self.__first_predict=False
        X_test=np.reshape(X_test,(1,-1))
        X_test = self.__poly.fit_transform(X_test)
        if self.__model != 0:
            Y = self.__model.predict(X_test)
            Y = list(map(int,Y[0]))
            return Y
        return 0

        # Using SVM
        # if self.__reg_type == 'S':
        #     svm = rg.SVM()
        #     return svm.predict(X_test)

        pass
