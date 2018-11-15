import numpy as np
import tensorflow as tf
import read_data as rd

def get_sub(data, filter, col):
	filtered = []
	for x in data:
		if(x[col] == filter):
			filtered.append(x)
	return filtered

def model():
	learn_rate = 0.001#need to determine an optimal learning rate
	batch_size = 1#need to determine a good batch size based on total dataset size
	epochs = 20#need to determine a good epoch amount
	lam = 0.0001#should vary this in order to determine impact on accuracy as well as fairness
	#procedure for choosing optimal lam for a dataset
	features, lables = rd.read_data()
	#shuffle data then take 80% for training 20% for validation
	print(features)

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
		with tf.name_scope("placeholder"):
			lam = tf.placeholder(tf.float32, shape=(1,1))

		with tf.name_scope("expected_loss_sub"):
			sub_loss = tf.placeholder(tf.float32, shape=(1,1))

		with tf.name_scope("expected_loss_ov"):
			ov_loss = tf.placeholder(tf.float32, shape=(1,1))

		with tf.name_scope("regularizer"):
			exp = tf.exp(sub_loss - ov_loss)
			reg = tf.log(1+exp)

		loss = tf.losses.mean_squared_error(labels, output) + lam * reg

	with tf.name_scope("optimizer"):
		opt = GradientDescentOptimizer(
			learning_rate=learn_rate).minimize(loss)


	with tf.Session as sess:
		sess.run(tf.initializers.global_variables())

		for batch in batches:
			for epoch in range(num_epochs):
				ov_loss = sess.run(loss, 
					feed_dict={x_features:batch["x"], labels:batch["y"]})
				sub_loss = sess.run(loss, 
					feed_dict={x_features:get_sub(batch["x"], group), 
					labels:get_sub(batch["y"])})

				sess.run(opt, 
					feed_dict={x_features:batch["x"], 
					labels:batch["y"], 
					sub_loss:group_loss, 
					ov_loss:overall_loss})

		#perform testing


		#initialize all variables
		#should include a tf writer to store results 
		#should include training logic
		#should include testing logic at the end
		pass

model()

