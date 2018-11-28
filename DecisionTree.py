from DecisionMethod import *
import numpy as np

class DecisionNode():
	def __init__(self, parent, attr_index, standard, is_continuous = False, threshold = None):
		self.attr_index = attr_index
		self.standard = standard
		self.sub_tree = {}
		self.is_continuous = is_continuous
		self.threshold = threshold
		self.parent = parent
		for e in standard:
			self.sub_tree[e] = []

	def __repr__(self):
		return 'attr_index: ' + str(self.attr_index) +'\nstandard: '+ str(self.standard) + '\nthreshold:' + str(self.threshold) +'\nsub_tree:\n' + str(self.sub_tree) + '\n\n'

class Leaf():
	def __init__(self, parent, label):
		self.label = label
		self.parent = parent

	def __repr__(self):
		return "Leaf Label: " + str(self.label) + '\n'

class DecisionTree():
	def __init__(self, data_set, spliter, max_depth, is_continuous, attribute_set= [0,1,2,3]):
		self.spliter = spliter
		self.max_depth = max_depth
		self.is_CART = isinstance(spliter, CART)
		#attribute_set = [i for i in range(data_set.getDimension())]
		self.root = self.__build(None, data_set, attribute_set, is_continuous, max_depth, 0)

	def __get_most_common_label(self, labels):
		return labels[np.argmax(np.unique(labels, return_counts=True)[1])]

	def __build(self, parent, data_set, attribute_set, is_continuous, max_depth, depth):
		#print(data_set)
		#print(attribute_set)
		#print('depth : ' + str(depth))
		Labels = data_set.getLabels()
		#print(Labels)
		#check the depth
		if (depth == max_depth):
			#print('reach max')
			return Leaf(parent, self.__get_most_common_label(Labels))

		#all the samples have the same label, return a leaf node, whose label is the most common label in samples
		if(len(set([label for label in Labels])) == 1):
			#print('labels are the same\n')
			return Leaf(parent, Labels[0])

		#print('attribute_values:', np.unique(data_set.getCols(attribute_set), axis = 0))
		
		#if the attribute_set is empty or all the attributes in samples have the same value return a leaf node, whose label is the most common label in sample space
		if(len(attribute_set) == 0 or np.unique(data_set.getCols(attribute_set), axis = 0).shape[0] == 1):
			#print('attribute_set may be emptyn\n')
			return Leaf(parent, self.__get_most_common_label(Labels))

		else:
			#print()
			threshold = 0
			#find the best split
			max_index, threshold = self.spliter(data_set, attribute_set, is_continuous)
			#print(max_index, threshold)
			data_split = self.spliter.data_split(data_set, max_index, is_continuous[max_index], threshold)
			#print(data_split)
			#split the data_set
			if(not is_continuous[max_index]):
				attribute_set.pop(max_index)
				
			attribute_value = [l for l in data_split]

			dNode = DecisionNode(parent, max_index, attribute_value, is_continuous[max_index], threshold)
			#print('attribute_value: ', attribute_value)
			#print('\n')

			for l in data_split:
				if(len(data_split[l]) != 0): 
					dNode.sub_tree[l] = self.__build(dNode.sub_tree[l], DataSet(data_set.extract(rows=data_split[l])), attribute_set, is_continuous, max_depth, depth + 1)
				else:
					dNode.sub_tree[l] = Leaf(dNode.sub_tree[l], self.__get_most_common_label(Labels))
 
			return dNode

	def __traverse(self, root, data):
		#print(root)
		if(isinstance(root, Leaf)):
			return root.label
		else:
			if(root.is_continuous):
				if(data[root.attr_index] <= root.threshold):
					return self.__traverse(root.sub_tree[LEFT_CHILD], data)
				else:
					return self.__traverse(root.sub_tree[RIGHT_CHILD], data)
			else:
				attr_value = data[root.attr_index]
				return self.__traverse(root.sub_tree[attr_value], data)	
				#print('attr_value: ', attr_value)
				#print()
		

	def __traverse_cart(self, root, data):
		#print(root)
		if(isinstance(root, Leaf)):
			return root.label
		else:
			if(root.is_continuous):
				if(data[root.attr_index] <= root.threshold):
					return self.__traverse_cart(root.sub_tree[LEFT_CHILD], data)
				else:
					return self.__traverse_cart(root.sub_tree[RIGHT_CHILD], data)
			else:
				if(data[root.attr_index] == root.threshold):
					return self.__traverse_cart(root.sub_tree[LEFT_CHILD], data)
				else:
					return self.__traverse_cart(root.sub_tree[RIGHT_CHILD], data)

	def predict(self, data):
		if(self.is_CART):
			return self.__traverse_cart(self.root, data)
		else:
			return self.__traverse(self.root, data)
			

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

if __name__ == '__main__':
	training_data = dataFile('zoo.csv', 16).Open()
	#print(training_data)
	training_data = DataSet(training_data)
	#print(training_data.getLabels())
	tree = DecisionTree(training_data, C4d5(), 10, [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],[i for i in range(0, training_data.getDimension())])
	#print(tree.predict(np.array(test_data)))