import numpy as np
import math
from DataSet import DataSet

LEFT_CHILD = 'left_child'
RIGHT_CHILD = 'right_child'

class DecisionMethod():
	#return selected(index) labels
	def ExtractLabels(self, label_set, index):
		"""
		array = [label_set[index[0]]]

		for x in range(1, len(index)):
			array.append(label_set[index[x]]) 
		"""
		#print(array)
		return np.array([label_set[i] for i in index])
		
	#return {attribute_val : [vector_index]}
	def data_split(self, data_set, attribute, is_continuous = False, threshold = None):
		data_split = {}
		if is_continuous:
			data_split[LEFT_CHILD] = []
			data_split[RIGHT_CHILD] = []

		for i in range(0, data_set.getNumberOfRow()):
			col = data_set.getCol(attribute)
			if is_continuous:
				if(col[i] <= threshold):
					data_split[LEFT_CHILD].append(i)
				else:
					data_split[RIGHT_CHILD].append(i)
			else:
				if(col[i] in data_split):		
					data_split[col[i]].append(i)
				else:
					data_split[col[i]] = [i]

		return data_split

class ID3(DecisionMethod):
	def CalEntropy(self, label_set):
		#print('label_set: ',label_set)
		label_val = set([x for x in label_set])
		ent = 0
		for label in label_val:
			p = float(label_set[label_set == label].shape[0]) / label_set.shape[0]
			log_p = math.log2(p)
			#print(p, log_p, p*log_p)
			ent -= p * log_p
		#print('entropy:' ,ent)
		return ent

	def __find_best_threshold(self, data_set, attribute, entropy):
		rows = data_set.getNumberOfRow()
		attribute_vals = np.sort(data_set.getCol(attribute))
		thresholds = [(attribute_vals[i] + attribute_vals[i + 1])/2 for i in range(0, attribute_vals.shape[0] - 1)]
		#print(thresholds)
		max_gain = -1
		best_thr = 0

		# print(thresholds)
		
		for thr in thresholds:
			gain = 0
			data_split = self.data_split(data_set, attribute, True, thr)
			#print(data_split)
			for l in data_split:
				if(data_split[l]):
					gain += self.CalEntropy(self.ExtractLabels(data_set.getLabels(), data_split[l])) * len(data_split[l]) / rows
			# print(gain)
			gain = entropy - gain
			#fucking python
			#fucking python
			if(gain < 1e-15):
				gain = 0
			#print('info_gain: ' , gain)
			if(max_gain < gain):
				#print(max_gain, gain)
				best_thr = thr
				max_gain = gain
		#print(max_gain, best_thr)
		return max_gain, best_thr

	def info_gain(self, data_set, attribute, entropy, is_continuous = False):
		#print(data_set)
		rows = data_set.getNumberOfRow()
		data = data_set.getData()
		data_split = {}
		calced_ent = 0

		if(is_continuous):
			max_gain, best_thr = self.__find_best_threshold(data_set, attribute, entropy)
			return max_gain, best_thr
		else:
			data_split = self.data_split(data_set, attribute)
			for l in data_split:
				calced_ent += self.CalEntropy(self.ExtractLabels(data_set.getLabels(), data_split[l])) * len(data_split[l]) / rows
			return (entropy - calced_ent), None

	#is_continuous is a list
	def __call__(self, data_set, attribute_set, is_continuous = None):
		entropy = self.CalEntropy(data_set.getLabels())
		#print('entropy :', entropy)
		#print()
		max_gain = -1
		res = 0,

		for i in range(0, len(attribute_set)):
			if(is_continuous is None):
				gain = self.info_gain(data_set, attribute_set[i], entropy)
			else:
				gain = self.info_gain(data_set, attribute_set[i], entropy, is_continuous[i])
				#print('gain:' , gain)
			# print('gain:' +str(gain))
			if max_gain < gain[0]:
				max_gain = gain[0]
				res = i, gain[1]
		return res

class C4d5(DecisionMethod):
	def __init__(self):
		self.gain = ID3()

	def IV(self, data_split, rows):
		res = 0
		#print(data_split)
		for d in data_split:
			p = len(data_split[d]) / rows
			res -= p * math.log2(p)
			#print(p, math.log2(p), res)
		return res

	def __call__(self, data_set, attribute_set, is_continuous = None):
		# print(data_set)
		# print(attribute_set)
		entropy = self.gain.CalEntropy(data_set.getLabels())
		# print(entropy)
		rows = data_set.getNumberOfRow()
		max_gain_ratio = -1
		res = 0,

		for i in range(0, len(attribute_set)):
			if(is_continuous is None):
				gain = self.gain.info_gain(data_set, attribute_set[i], entropy) 
			else:
				gain = self.gain.info_gain(data_set, attribute_set[i], entropy, is_continuous[i])
			#print('gain:', gain)
			#print(attribute_set[i])
			if(gain[0] != 0):
				gain_ratio = gain[0] / self.IV(self.data_split(data_set, attribute_set[i]), rows)
			else:
				gain_ratio = 0
			# print(gain_ratio)
			if(max_gain_ratio < gain_ratio):
				max_gain_ratio = gain_ratio
				res = i, gain[1]

		# print('res:', res)
		return res


class CART(DecisionMethod):
	def data_split(self, data_set, attribute, is_continuous = False, threshold = None):
		data_split = {}
		data_split[LEFT_CHILD] = []
		data_split[RIGHT_CHILD] = []

		for i in range(0, data_set.getNumberOfRow()):
			col = data_set.getCol(attribute)
			if is_continuous:
				if(col[i] <= threshold):
					data_split[LEFT_CHILD].append(i)
				else:
					data_split[RIGHT_CHILD].append(i)
			else:
				if(col[i] == threshold):
					data_split[LEFT_CHILD].append(i)
				else:
					data_split[RIGHT_CHILD].append(i)

		return data_split

	def Gini(self, label_set):
		#print(label_set)
		impurity = 1
		label_val = set([x for x in label_set])
		for label in label_val:
			p = float(label_set[label_set == label].shape[0]) / label_set.shape[0]
			#print(p)
			impurity -= p**2
			#print(impurity)
		return impurity

	def __find_best_val(self, data_set, attribute, is_continuous = False):
		rows = data_set.getNumberOfRow()
		if(is_continuous):
			attribute_vals = np.sort(data_set.getCol(attribute))
			standard = [(attribute_vals[i] + attribute_vals[i + 1])/2 for i in range(0, attribute_vals.shape[0] - 1)]
		else:
			standard = np.unique(data_set.getCol(attribute))

		best_val = 0
		min_gini_index = 1

		for val in standard:
			gini_index = 0
			data_split = self.data_split(data_set, attribute, is_continuous, val)
			#print(is_continuous, data_split)
			for d in data_split:
				if(data_split[d]):
					gini_index += self.Gini(self.ExtractLabels(data_set.getLabels(), data_split[d])) * len(data_split[d]) / rows
			#print('gini_index: ', gini_index)
			if(gini_index < min_gini_index):
				best_val = val
				min_gini_index = gini_index

		#print(is_continuous, best_val)
		return min_gini_index, best_val


	def __call__(self, data_set, attribute_set, is_continuous = None):
		min_gini_index = 1

		best_index = 0
		best_val = 0
		gini_index = 0
		#print(is_continuous)
		for i in range(0, len(attribute_set)):
			if(is_continuous is not None):
				gini_index, val = self.__find_best_val(data_set, attribute_set[i], is_continuous[i])
			else:
				gini_index, val = self.__find_best_val(data_set, attribute_set[i])

			#print(impurity, best_val, is_continuous[i])

			if(gini_index < min_gini_index):
				best_index = attribute_set[i]
				min_gini_index = gini_index
				best_val = val
				#print('best_index and impurity ',best_index, impurity)

		return best_index, best_val
		

if __name__ == '__main__':
	training_data = [
    [1, 1, 3, 1],
    [1, 2, 3, 1],
    [1, 3, 3, 2],
    [1, 3, 3, 2],
	]

	#data = [d for d in training_data]
	data_set = DataSet([d for d in training_data])
	#print(data_set)
	#print(data_set.getData())
	#print(data_set.getCol(0))
	#print(data_set.getCorrespondingLabel(1))
	#print(data_set.getDimension())
	gain = C4d5()
	#print(gain.CalEntropy(data_set.getLabels()))
	print(gain(data_set, [0,1], [True, True]))