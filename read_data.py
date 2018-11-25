import tensorflow as tf
import numpy as np

def get_line(f):
	s = f.read(130)

	list_vals = []
	for i in range(15, 21, 2):
		list_vals.append(float(s[i:i+2]))

	for i in range(23, 41, 2):
		list_vals.append(float(s[i:i+2]))

	list_vals.append(float(s[58:60]))

	list_vals.append(float(s[63:65]))
	list_vals.append(float(s[65:68]))

	list_vals.append(float(s[70:72]))
	list_vals.append(float(s[72:74]))

	list_vals.append(float(s[80:82]))
	list_vals.append(float(s[82:84]))

	list_vals.append(float(s[86:88]))
	list_vals.append(float(s[88:90]))

	list_vals.append(float(s[94:96]))
	list_vals.append(float(s[96:98]))

	list_vals.append(float(s[100:102]))

	for i in range(102, 120):
		list_vals.append(float(s[i]))

	list_vals.append(float(s[120:122]))

	for i in range(124, 130, 2):
		list_vals.append(float(s[i:i+2]))

	if (int(list_vals[11]) > 0):
	#if crime commited put 1 if unknown put 0 and if
	#no crime commited put -1 
		y_vals = 1.0 
	elif (int(list_vals[11]) == 0):
		y_vals = -1.0 
	else:
		y_vals = 0.0

	if (int(list_vals[2]) == 4):
	#if race is black set as 1 else zero
		list_vals[2] = 1.0 
	else:
		list_vals[2] = 0.0 

	list_vals = list_vals[0:11] + list_vals[12:]
	return y_vals, list_vals, s

def read_data(filename=None, num_samples=20000, save_loc="new_dataset.ascii"):
	test_size = int(num_samples / 4)
	x = []
	y = []
	test_x = []
	test_y = []
	f_two = open(save_loc, "w+")

	if filename == None:
		x = np.array([[1.0, 2.0, 2.0],[3.0, 5.0, 7.0]])
		y = np.array([[1.0],[0.0]])
		return x, y
	else:
		#file = np.loadtxt(filename, dtype=np.float32)
		with open(filename) as f:
			sample = 0
			while sample < num_samples:
				y_smp, x_smp, s = get_line(f)
				f.read(1)
				if(y_smp != 0):
					print("wrote a value")
					f_two.write(s)
					x.append(np.array(x_smp))
					y.append(np.array(y_smp))
					sample += 1
			sample = 0
			while sample < test_size:
				y_smp, x_smp, s = get_line(f)
				f.read(1)
				if(y_smp != 0):
					test_x.append(np.array(x_smp))
					test_y.append(np.array(y_smp))
					sample += 1
		y = np.reshape(y, (-1, 1))
		test_y = np.reshape(test_y, (-1, 1))
		f_two.close()
		return x, y, test_x, test_y


def get_groups(data, filter_value, col):
	filteredx = []
	filteredy = []
	rest_x = []
	rest_y = []
	x = data["x"]
	y = data["y"]
	epsilon = 0.000001

	for i in range(len(x)):
		if(np.abs(x[i][col]) - filter_value) < epsilon:
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

read_data(filename="TEDS-D-2006-2011-DS0001-data/TEDS-D-2006-2011-DS0001-data-ascii.txt")
