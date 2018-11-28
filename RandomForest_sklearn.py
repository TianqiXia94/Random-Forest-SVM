import numpy as np
from DataSet import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import CV

class dataFile :
    address = ''
    data = []
    label = 0
    def __init__(self, item, label):
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

            datasetTemp.remove(y)

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
		
def main():
    fold = 10
    data_set = dataFile('wine/wine.data', 0).Open()
    cv_data = CV.CrossValidation(data_set, fold)
    acc = []
    res = 0
    for i in  range(5):		
        for i in range(0, fold):
            training_data, test_data = cv_data.findTest(i)
    
            training_data = DataSet(training_data)
            test_data = DataSet(test_data)
    
            size_of_test_data = test_data.getNumberOfRow()
    
            X_train, X_test, y_train, y_test = training_data.getData(), test_data.getData(), training_data.getLabels(), test_data.getLabels()
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_predict, normalize = False) / size_of_test_data
            acc.append(accuracy)
    
        print(1 - cv_data.Accuracy(acc))
        res += 1 - cv_data.Accuracy(acc)
    print(res / 5)
        




	
if __name__ == '__main__':
	main()
	


