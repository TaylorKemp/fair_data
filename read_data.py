import tensorflow as tf
import numpy as np

def read_csv(filename=None):
	if filename == None:
		return

	df = np.genfromtxt(filename, delimiter=',')
	#data = np.genfromtxt(filename, delimiter=',')

	df = np.delete(df, 0, 0)
	np.random.shuffle(df)
	train = int(len(df) * .8)
	y_values = np.reshape(df[:train,0], (-1, 1))
	y_values[np.where(y_values == 0)] = -1.0
	x_values = df[:train, 1:]
	y_test = np.reshape(df[train:, 0], (-1, 1))
	y_test[np.where(y_test == 0)] = -1.0
	x_test = df[train:,1:]

	return x_values, y_values, x_test, y_test


def get_groups(data, filter_value, col):
	filteredx = []
	filteredy = []
	rest_x = []
	rest_y = []
	x = data["x"]
	y = data["y"]
	epsilon = 0.000001

	for i in range(len(x)):
		if(x[i][col] == filter_value):
			filteredx.append(x[i])
			filteredy.append(y[i])
		else:
			rest_x.append(x[i])
			rest_y.append(y[i])

	return filteredx, filteredy, rest_x, rest_y

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

#clean_dataset(filename="TEDS-D-2006-2011-DS0001-data/TEDS-D-2006-2011-DS0001-data-ascii.txt")