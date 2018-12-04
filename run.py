import tensorflow as tf
import numpy as np
import main as mn
import read_data as rd
import matplotlib.pyplot as plt

#overall accuracy vs fairness(african vs other)


def run_multiple():
	lambdas = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0]

	acc_afr = ()
	acc_other = ()
	acc_overall = ()

	for lam in lambdas:
		if(lam == 0.001):
			weights, bias, test_labels, test_features = mn.model(lam, True)
		else:
			weights, bias, test_labels, test_features = mn.model(lam, True)

		weights = np.copy(weights)
		bias = np.copy(bias)
		test_labels = np.copy(test_labels)
		test_features = np.copy(test_features)

		test_labels	= np.reshape(test_labels, (-1, 1))
		afr_feat, afr_lab, rest_feat, rest_lab = rd.get_groups({"x":test_features, "y":test_labels}, 1, 4)

		acc_afr += (get_accuracy(afr_feat, weights, bias, afr_lab), )
		acc_other += (get_accuracy(rest_feat, weights, bias, rest_lab), )
		acc_overall += (get_accuracy(test_features, weights, bias, test_labels), )

	plot_values(acc_afr, acc_other, acc_overall)

def get_accuracy(x, w, b, y):
	out_val = np.matmul(x, w) + b

	num_correct = 0

	out_val = np.reshape(out_val, (1, -1))
	test_labels = np.reshape(y, (1, -1))

	for i in range(len(test_labels[0])):
		if(out_val[0][i] > 0 and test_labels[0][i] == 1):
			num_correct += 1
		elif(out_val[0][i] <= 0 and test_labels[0][i] == -1):
			num_correct += 1

	return float(num_correct) / len(y) * 100


def plot_values(afro, rest, oval):
	#afro = (60.62, 57.10, 60.86, 59.59, 60.16, 54.12, 44.67, 45.88, 48.41, 44.61)
	#rest = (66.10, 68.91, 64.86, 67.78, 58.87, 41.72, 40.87, 41.72, 38.35, 38.60)
	#oval = (63.24, 62.75, 62.83, 63.56, 59.51, 48.18, 42.83, 43.89, 43.48, 41.62)

	ind = np.arange(len(afro))  # the x locations for the groups
	width = 0.3  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind - width, afro, width, color='SkyBlue', label='African-American')
	rects2 = ax.bar(ind, rest, width, color='IndianRed', label='other')
	rects3 = ax.bar(ind + width, oval, width, color='green', label='overall')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Accuracy(%)')
	ax.set_xlabel('Regularization Parameter(Fairness)')
	ax.set_title('Affect of Regularization Parameter on Accuracy')
	ax.set_xticks(ind)
	ax.set_xticklabels(('0.0', '0.001', '0.01', '0.1', '0.5', '1.0', '2.0', '4.0', '5.0', '10.0'))
	ax.legend()

	plt.show()

	print("done")


run_multiple()
