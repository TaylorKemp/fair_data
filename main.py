import numpy as np
import tensorflow as tf
import read_data as rd

def model():
	learn_rate = 0.001#need to determine an optimal learning rate
	batch_size = 1#need to determine a good batch size based on total dataset size
	num_epochs = 20#need to determine a good epoch amount
	num_groups = 2
	lam = 0.0001#should vary this in order to determine impact on accuracy as well as fairness
	groups = [0, 1]# should have values of group
	column = 2#will need to obtain from dataset
	#procedure for choosing optimal lam for a dataset
	features, labels = rd.read_data()
	batches = rd.get_batches(features, labels, batch_size)
	#shuffle data then take 80% for training 20% for validation

	num_features = len(features[0])#determine num_features based on 

	with tf.name_scope("features"):
		x_features = tf.placeholder(tf.float32, shape=(batch_size, num_features))

	with tf.name_scope("labels"):
		labels = tf.placeholder(tf.float32, shape=(batch_size, 1))

	with tf.name_scope("model"):
		weights = tf.Variable(tf.random_normal([num_features, 1]))
		bias = tf.Variable(0.0)
		output = tf.matmul(x_features, weights) + bias
		weight_summary = tf.summary.histogram("weight_summary", weights)
		bias_summary = tf.summary.histogram("bias_summary", bias)

	with tf.name_scope("loss_function"):
		with tf.name_scope("placeholder"):
			lam_value = tf.placeholder(tf.float32, shape=(1,1))

		with tf.name_scope("expected_loss_sub"):
			sub_loss = tf.placeholder(tf.float32)

		with tf.name_scope("expected_loss_ov"):
			ov_loss = tf.placeholder(tf.float32, shape=(batch_size,1))

		with tf.name_scope("regularizer"):
			exp = tf.exp(sub_loss - ov_loss)
			reg = tf.log(1+exp)

		squared_error = tf.losses.mean_squared_error(labels, output)
		loss = squared_error + lam_value * reg
		tf.summary.scalar("loss_summary", loss)

	with tf.name_scope("optimizer"):
		opt = tf.train.GradientDescentOptimizer(
			learning_rate=learn_rate).minimize(loss)


	with tf.Session() as sess:
		sess.run(tf.initializers.global_variables())
		summary = tf.summary.merge_all()
		writer = tf.summary.FileWriter("graph", sess.graph)

		for batch in batches:
			for epoch in range(num_epochs):
				sub_x, sub_y = rd.get_sub(batch, groups[epoch % num_groups], column)

				ov_loss = sess.run(squared_error, 
					feed_dict={x_features:batch["x"], labels:batch["y"]})
				
				print(sub_x)
				print(sub_y)
				sub_loss = sess.run(squared_error, 
					feed_dict={x_features:sub_x, 
					labels:sub_y})

				summary, _ = sess.run([summary, opt], 
					feed_dict={x_features:batch["x"], 
					labels:batch["y"], 
					sub_loss:group_loss, 
					ov_loss:overall_loss})
				if(epoch == num_epochs - 1):
					writer.add_summary(summary)

model()
