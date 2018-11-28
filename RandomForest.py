import random
from DecisionTree import *
import numpy as np
import copy
from sklearn import datasets
from sklearn.model_selection import train_test_split
import CV

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
                self.data.append(line.strip())

        for i in range(0, len(self.data)) :
            dataset.append(self.data[i].split(","))
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


class Bootstrap(object):
    def __init__(self):
        pass

    def __init__(self, dataset,m):
        self.dataset = dataset
        self.m = m
        # self.dataset = DataSet(dataset).getData()
        # self.label = DataSet(dataset).getLabels()

    def bootstrap(self):
        dataDict = []
        datasetTemp = [0] * len(self.dataset)
        for j in range(0, len(self.dataset)):
            # dataDict[j] = []
            for i in range(0, len(self.dataset)):
                datasetTemp[i] = self.dataset[random.randrange(0, len(self.dataset))]
            dataDict.append(copy.deepcopy(datasetTemp))

        return dataDict


class Bagging(Bootstrap):
    def __init__(self):
        super(Bootstrap, self).__init__(self)

    def __init__(self, m, dataset, feature):
        Bootstrap.__init__(self, dataset, m)
        
        self.feature = int((feature-1) / 1)

    def bagging(self):
        bootstrap = self.bootstrap()
        feature_num = [0] * (len(self.dataset[0])-1)
        feature = []
        for i in range(0, len(feature_num)):
            feature_num[i] = i
        # print(feature_num)
        for i in range(len(self.dataset)):
            featureTemp = random.sample(feature_num, self.feature)
            featureTemp.sort()
            feature.append(featureTemp)
            bootstrap[i] = [bootstrap[i], featureTemp]
        # list(set(bootstrap.values()))
        # print(bootstrap)

        # delet the repeat object
        the_list = []
        for level in bootstrap:
            if level not in the_list:
                the_list.append(level)
        # print(the_list)
        # print(the_list)
        for i in range(len(bootstrap)):
            if i < len(the_list):
                bootstrap[i] = the_list[i][0]
                feature[i] = the_list[i][1]
            else:
                bootstrap.pop()
                feature.pop()

        return bootstrap, feature


class RandomForest(Bagging, DecisionTree):
    def __init__(self, m, feature, data_set, spliter, max_depth, is_continuous):
        Bagging.__init__(self, m, data_set,feature)
        #DecisionTree.__init__(self, data_set, spliter, max_depth, is_continuous, attribute_set)
        self.spliter = spliter
        self.max_depth = max_depth
        self.is_continuous = is_continuous
        # attribute_set = [i for i in range(data_set.getDimension())]
        bagging = Bagging(m,data_set,feature)
        data, attribute_set = bagging.bagging()
        self.trees = [0] * len(data_set)
        for i in range(0, len(data)):
            DecisionTree.__init__(self, DataSet(data[i]), spliter, max_depth, is_continuous, attribute_set[i])
            self.trees[i] = self.root
        #return self.trees

    '''def vote(self):
        return 0'''

    def vote(self,data):
        vote = [0]*len(self.trees)
        for i in range(0, len(self.trees)):
            self.root = self.trees[i]
            vote[i] = self.predict(data)
        return vote[np.argmax(np.unique(vote, return_counts=True)[1])]
    '''def RandomForest(self):
        #bagging = Bagging(data_set, len(data_set[0]))
        #data, f = bagging.bagging()

        for i in range(0, len(self.data)):
            self.trees[i] = self.__build(None, self.data[i], self.attribute_set[i], self.is_continuous, self.max_depth, 0)
        return self.trees
'''

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

    data = dataFile('iris.csv', 4)
    training_data_wine = data.Open()
    #print(training_data_wine)
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

    if i == 1:
        return training_data,test_data
    if i == 2:
        return training_data_1,test_data_1
    if i == 3:
        return training_data_wine,training_data_wine
    if i == 4:
        return X_train,X_test


if __name__ == '__main__':

    training_dataset, test_data = test(3)
    count = 0
    count1 = 0
    forests = []
    m_tree = 10
    bestForest_label = 0
    miniAccuracy = 0
    bestForest = [0]
    accuracy_F = []
    accuracy_T = []
    depth = 10
    #print(training_dataset)
    for m_tree in range(10, 11):
        cv_data = CV.CrossValidation(training_dataset, 10)
        for i in range(10):
            training_data, test_data = cv_data.findTest(i)
            #tree = DecisionTree(DataSet(training_data), C4d5(), 10, [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],[i for i in range(0, 16)])
            # bagging = Bagging(training_data, len(training_data[0]))
            # data, f = bagging.bagging()
            trees = RandomForest(m_tree, 4, training_data, CART(), depth,
                                 [True, True, True, True])
			
            forests.append(trees.trees)
            # print(tree.predict(np.array(test_data)))
            # trees.RandomForest()
            '''if len(data) < 5:
                print("repeat", len(data), data, len(f), f)
                count = count + 1'''
            # print(data,f)
            # print(trees)
            # print(trees.vote(np.array(test_data_1)))
            Y_label = len(test_data[0]) - 1
            for j in range(len(test_data)):
                # print(trees.vote(np.array(test_data[j])),test_data[j][4])
                if trees.vote(np.array(test_data[j])) == test_data[j][Y_label]:
                    count = count + 1

            print("Forest", count, len(test_data), count / len(test_data))
            accuracy_F.append((count / len(test_data)))
            if (count / len(test_data)) > miniAccuracy:
                miniAccuracy = (count / len(test_data))
                bestForest_label = i
                bestForest = trees.trees

            count = 0
            count1 = 0
        print("m_tree", m_tree, "error rate", 1 - cv_data.Accuracy(accuracy_F))

        print("")

        accuracy_T.clear()
        accuracy_F.clear()

    print("end")