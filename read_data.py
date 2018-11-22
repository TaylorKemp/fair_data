import tensorflow as tf
import numpy as np

def read_data():
	x = np.array([[1.0, 2.0, 2.0],[3.0, 5.0, 7.0]])
	y = np.array([[1.0],[0.0]])
	return x, y

def get_sub(data, filter_value, col):
	filteredx = []
	filteredy = []
	x = data["x"]
	y = data["y"]
	epsilon = 0.000001

	for i in range(len(x)):
		if(np.abs(x[i][col]) - filter_value) < epsilon:
			filteredx.append(x[i])
			filteredy.append(y[i])

	return filteredx, filteredy

def get_batches(features, labels, batch_size):
	num_batches = int(len(features) / batch_size)
	batches = []
	lower = 0
	upper = batch_size
	if(len(features) <= batch_size):
		return [{"x":features, "y":labels}]

	for batch in range(num_batches):
		batches.append({"x":features[lower:upper], "y":labels[lower:upper]})
		lower = upper
		upper += batch_size

	#last batch not getting added
	return batches
