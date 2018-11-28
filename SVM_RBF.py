import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import CV
from DataSet import DataSet

class dataFile :
    address = ''
    data = []
    label = 0
    def __init__(self, item,label):
        self.address = item
        self.label = label

    def Open(self):
        dataset = []
        datasetTemp = [0]
        y = 0
        with open(self.address, 'r') as file:
            for line in file.readlines() :
                #print(line)
                self.data.append(line.strip())
        #print(self.data)
        for i in range(0, len(self.data)) :
            dataset.append(self.data[i].split(","))
            #print(dataset)
            for j in range (0, len(dataset[i])):
                if j==self.label :
                    y = dataset[i][j]
                datasetTemp = dataset[i]

            del datasetTemp[self.label]

            for j in range (0, len(datasetTemp)+1):
                if j == len(datasetTemp):
                    datasetTemp.append(y)
                try :
                    datasetTemp[j] = int(datasetTemp[j])
                except:
                    try:
                        datasetTemp[j] = float(datasetTemp[j])
                    except:
                        datasetTemp[j] = datasetTemp[j]


            dataset[i] = datasetTemp
        #str = "str"
        #print(float(str))
        '''if type(dataset[0][0][1]) is int :

            print('true int')'''
        #print(dataset)
        return dataset

def test(i):
    training_data = [
        ['Green', 3, 1],
        ['Yellow', 3, 1],
        ['Red', 1, 2],
        ['Red', 1, 2],
        ['Yellow', 3, 3],
    ]
    test_data = ['Yellow', 3, 3]

    training_data_1 = [
        [1, 3, 3, 1],
        [2, 3, 2, 3],
        [3, 1, 1, 2],
        [3, 1, 1, 2],
        [2, 3, 1, 1],
        [2, 3, 3, 2],
        [2, 2, 2, 1],
        [2, 1, 3, 1],
    ]
    test_data_1 = [3, 1, 1, 2]

    X_train_wine = []
    Y_train_wine = []
    X_test_wine = []
    Y_test_wine = []
    data = dataFile('D:/Tianqi_Xia/UoM/COMP61011-Foundations_of_Machine_Learning/dataset/wine/wine.data', 0)
    training_data_wine = data.Open()
    #print("training_data_wine",training_data_wine)
    '''cv_data = CV.CrossValidation(training_data_wine,2)
    training_data, test_data = cv_data.findTest(1)
    for j in range(len(training_data)):
        X_train_wine.append(training_data[j])
        Y_train_wine.append(X_train_wine[j].pop())
    for j in range(len(test_data)):
        X_test_wine.append(test_data[j])
        Y_test_wine.append(X_test_wine[j].pop())
    #print(training_data_wine)'''
    test_data_wine = []

    iris = datasets.load_iris()
    X = list(iris.data[:, :4])
    Y = list(iris.target)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    for j in range(len(X_train)):
        X_train[j] = X_train[j].tolist()
        X_train[j].append(Y_train[j])
    for j in range(len(X_test)):
        X_test[j] = X_test[j].tolist()
        X_test[j].append(Y_test[j])
    #print(X_train,X_test)
    for j in range(len(X_test)):
        X_train.append(X_test[j])
    print(len(X_train))

    if i == 1:
        return training_data,test_data
    if i == 2:
        return training_data_1,test_data_1
    if i == 3:
        return training_data_wine#np.array(X_train_wine),np.array(Y_train_wine),np.array(X_test_wine),np.array(Y_test_wine)
    if i == 4:
        return X_train

def svm():

    training_dataset = test(3) #X_train, Y_train, X_test, Y_test
    C_range = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1,1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    gamma_range = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
                   1e8]
    count = 0
    mini_error = 1
    best_C = 0
    best_gamma = 0
    k_fold = 10
    accuracy_C = []
    accuracy_gamma = []
    '''iris = datasets.load_iris()
    X = list(iris.data[:, :4])
    Y = list(iris.target)'''

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    cv_data = CV.CrossValidation(training_dataset, k_fold)
    for C in C_range:
        for i in range(k_fold):
            training_data, test_data = cv_data.findTest(i)
            data = DataSet(training_data)
            t_data = DataSet(test_data)

        #classifiers = []


        #for gamma in gamma_range:
            clf = SVC(C=C, gamma=1e-7, decision_function_shape='ovo')
            clf.fit(data.getData(), data.getLabels())
            #classifiers.append((C, 1e-7, clf))
            #print(classifiers)
            result = clf.predict(t_data.getData())
            #print(result)
            for i in range(len(t_data.getData())):
                if result[i] == t_data.getLabels()[i]:
                    count = count + 1
            #print(count / len(t_data.getData()))
            accuracy_C.append((count / len(t_data.getData())))
            if (1-(count / len(t_data.getData())))<mini_error:
                mini_error = 1-(count / len(t_data.getData()))
                best_C = C
                best_gamma = 1e-7

            count = 0

        print("C","error rate", 1 - cv_data.Accuracy(accuracy_C),"C",C)
        #for C in C_range:
        accuracy_C.clear()

    for gamma in gamma_range:
        for i in range(k_fold):
            training_data, test_data = cv_data.findTest(i)
            data = DataSet(training_data)
            t_data = DataSet(test_data)
            clf = SVC(C=1000000, gamma=gamma, decision_function_shape='ovo')
            clf.fit(data.getData(), data.getLabels())
            #classifiers.append((1000000, gamma, clf))
            #print(classifiers)
            result = clf.predict(t_data.getData())
            #print(result)
            for i in range(len(t_data.getData())):
                if result[i] == t_data.getLabels()[i]:
                    count = count + 1
            #print(count / len(t_data.getData()))
            accuracy_gamma.append((count / len(t_data.getData())))
            if (1-(count / len(t_data.getData())))<mini_error:
                mini_error = 1-(count / len(t_data.getData()))
                best_C = 1000000
                best_gamma = gamma

            count = 0

        print("gamma","error rate", 1 - cv_data.Accuracy(accuracy_gamma), "gamma", gamma)
        accuracy_gamma.clear()
'''    for i in range(len(classifiers)):
        result = classifiers[i][2].predict(X_test)
        print(result)
        for i in range(len(X_test)):
            if  result[i] == Y_test[i]:
                count = count+1
        print(count/len(X_test))
        count = 0'''

if __name__ == "__main__":
    svm()