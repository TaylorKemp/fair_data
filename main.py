import numpy as np
import tensorflow as tf

def read_data():
	return 0, 0

def model():
	learn_rate = 0.001#need to determine an optimal learning rate
	lam = 0.0001#should very this in order to determine impact on accuracy as well as fairness
	#procedure for choosing optimal lam for a dataset
	features, lables = read_data()
	#shuffle data then take 80% for training 20% for validation

	num_features = len(features[0])#determine num_features based on 

	with tf.name_scope("features"):
		x_features = tf.placeholder(tf.float32, shape=(batch_size, num_features))

	with tf.name_scope("labels"):
		labels = tf.placeholder(tf.float32, shape=(batch_size, 1))

	with tf.name_scope("model"):
		weights = tf.Variable(tf.random_normal([num_features, 1]))
		bias = tf.Variable(0.0)
		output = x_features * weights + bias

	with tf.name_scope("loss_function"):
		reg = tf.math.exp()
		loss = tf.losses.mean_squared_error(labels, output) + reg#logic for loss function goes here

	with tf.name_scope("optimizer"):
		opt = GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)


	with tf.Session as sess:
		sess.run(tf.initializers.global_variables())

		for batch in batches:
			sess.run(opt)

		#perform testing


		#initialize all variables
		#should include a tf writer to store results 
		#should include training logic
		#should include testing logic at the end
		pass
