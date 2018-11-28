
import numpy as np
import random
from sklearn import datasets
import copy

class CrossValidation():
    def __init__(self, data, cv = 10):
        self.data = copy.deepcopy(data)
        self.cv = int(cv)
        self.cv_dataTemp = self._K_fold()
        self.accuracy = 0.0
        self.average = 0.0

    def _K_fold(self):
        CV_dataTemp = []
        #CV_data = []
        size = len(self.data)
        remainder = size % self.cv
        dataTemp = []
        dataRemainder = []
        random.shuffle(self.data)
        #print(self.data)
        if remainder == 0:
            for i in range(self.cv) :
                for j in range(int(size/self.cv)):
                    dataTemp.append(self.data.pop())#.tolist())
                CV_dataTemp.append(copy.deepcopy(dataTemp))
                dataTemp.clear()
        else :
            for i in range(remainder):
                dataRemainder.append(self.data.pop())#.tolist())
            CV_dataTemp = self._K_fold()

            for i in range(remainder):
                CV_dataTemp[i].append(dataRemainder[i])
                #print(len(CV_dataTemp[i]))
        #print(len(CV_dataTemp))

        return CV_dataTemp

    def findTest(self,k):
        cv_train = []
        cv_test = []
        for index in range(len(self.cv_dataTemp)):
            if index == k :
                cv_test = self.cv_dataTemp[index]

            else :
                for i in range(len(self.cv_dataTemp[index])):
                    cv_train.append(self.cv_dataTemp[index][i])
        return cv_train,cv_test

    def Accuracy(self, accuracy):
        self.accuracy = 0
        self.average = 0
        print(accuracy)
        for i in range(len(accuracy)):
            self.accuracy = self.accuracy + accuracy[i]
        self.average = self.accuracy/self.cv

        return self.average



if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    Y = iris.target
    k_fold = 15
    cv_data = CrossValidation(list(X),k_fold)
    for i in range(k_fold):
        print(cv_data.cv_dataTemp)
        cv_train, cv_test = cv_data.findTest(i)
        print(len(cv_train), len(cv_test))