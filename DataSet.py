import numpy as np

class DataSet():
    def __init__(self):
        pass

    def __init__(self, data):
        self.all = np.array(data)

    def __repr__(self):
        return 'data:\n' + str(self.all[:,:-1]) + '\n' + 'labels: ' + str(self.all[:,-1]) + '\n'

    def getAll(self):
        return self.all

    def getDimension(self):
        return self.all.shape[1] - 1

    def getNumberOfRow(self):
        return self.all.shape[0]

    def getData(self):
        return self.all[:,:-1]

    def getLabels(self):
        return self.all[:,-1]

    def getCol(self, col):
        return self.all[:,col]

    def getCols(self, cols):
        return self.all[:,np.array(cols)]

    def getDtype(self):
        return self.all.dtype

    def getItem(self, row, col):
        return self.all[row][col]

    def extract(self,**kw):
        if('rows' not in kw):
            rows = [i for i in range(0, self.getNumberOfRow())]
        else:
            rows = kw['rows']

        if('cols' not in kw):
            cols = [i for i in range(0, self.getDimension() + 1)]
        else:
            cols = list(kw['cols'])
            cols.append(self.getDimension())

        return self.all[np.array(rows)[:, None], np.array(cols)]

    def delete(self, **kw):
        if 'rows' in kw:
            rows = kw['rows']
            total = [i for i in range(0, self.getNumberOfRow())]
            rows = list(set(total).difference(set(rows)))
        else:
            rows =  [i for i in range(0, self.getNumberOfRow())]

        if 'cols' in kw:
            cols = kw['cols']
            total = [i for i in range(0, self.getDimension() + 1)]
            cols = list(set(total).difference(set(cols)))
        else:
            cols = [i for i in range(0, self.getDimension() + 1)]
        #print(cols)
        return self.all[np.array(rows)[:, None], np.array(cols)]

    def fix(self):
        b = self.all[:, 1:]
        self.all = np.concatenate((b, np.array([self.all[:, 0]]).T), axis=1)
        return self.all


if __name__ == '__main__':
    training_data = [
    ['Green', 3, 1],
    ['Yellow', 3, 1],
    ['Red', 1, 2],
    ['Red', 1, 2],
    ['Yellow', 3, 3],
    ]

    data = [d for d in training_data]
    data_set = DataSet(data)
    #print(data_set)
    #print(data_set.getData())
    print(np.sort(data_set.getCol(0)))
    #print(data_set.getDimension())
    #print(data_set.extract(cols=[0,1]))
    #print(data_set.getLabels())
    #print(data_set.delete(cols = [1]))
    #print(data_set.getCols([0,1]))