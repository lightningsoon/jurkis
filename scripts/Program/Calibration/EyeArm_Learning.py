# -*- coding=utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import random

# from calibrate import ArmEye_collectingData
# a=ArmEye_collectingData()
#
# print(len(list(a.para)))
class Pinocchio(object):
    def __init__(self):
        self._reg_type = 'P'  # 回归器类型
        self._first_predict=True
        self.__poly = preprocessing.PolynomialFeatures(degree=4)
        self.model_name='/home/momo/Project/jurkis_ws/src/jurkis/scripts/Program/Calibration/Model/Poly/model.m'
        if self._reg_type == 'P' and self._first_predict:
            self._model = self._restoreModel()
            self._first_predict=False
        super(Pinocchio,self).__init__()
    def train(self):
        self.readData()
        X,Y=np.array(self.X),np.array(self.Y)
        L=np.arange(len(X))
        random.shuffle(L)
        X,Y=X[L],Y[L]
        # XY=np.array(zip(self.X,self.Y))
        # random.shuffle(XY)
        # Training neural network
        # if self._reg_type == 'N':
        #     net = rg.Net()
        #     net.train(X, Y, learnrate=0.001, max_iter=10000)

        # Training polynomial regression model
        if self._reg_type == 'P':
            
            X = self.__poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X, Y)
            self.__dumpModel__(model)
            self._model=model
            self.test(X, Y)
            return 1

        # Training SVM model
        # if self._reg_type == 'S':
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
        print ("score:", self._model.score(x, y))
        pass

    def __dumpModel__(self, model):
        if self._reg_type == 'P':
            joblib.dump(model, self.model_name)
        return 1
        pass

    def _restoreModel(self):
        try:
            model = joblib.load(self.model_name)
        except:
            print("Import model error! NO %s" % (self.model_name))
            exit()
        return model
        pass

    def predict(self, X_test):
        # X_test：int[3] (3,)
        # Y:int[4] (4,)
        # Using neural network model
        # if self._reg_type == 'N':
        #     net = rg.Net()
        #     return net.predict(X_test)

        # Using polynomial regression model
        
        X_test=np.reshape(X_test,(1,-1))
        X_test = self.__poly.fit_transform(X_test)
        if self._model != 0:
            Y = self._model.predict(X_test)
            Y = list(map(int,Y[0]))
            return Y
        return 0

        # Using SVM
        # if self._reg_type == 'S':
        #     svm = rg.SVM()
        #     return svm.predict(X_test)

        pass
