import numpy as np
import tensorflow as tf
import read_data as rd

def model():
	tf.summary.FileWriterCache.clear()
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

	with tf.name_scope("subGroup"):
		group_x = tf.placeholder(tf.float32, shape=(None, num_features))
		group_y = tf.placeholder(tf.float32, shape=(None, 1))

	with tf.name_scope("remainingData"):
		rest_of_x = tf.placeholder(tf.float32, shape=(None, num_features))
		rest_of_y = tf.placeholder(tf.float32, shape=(None, 1))

	with tf.name_scope("features"):
		x_features = tf.placeholder(tf.float32, shape=(None, num_features))

	with tf.name_scope("labels"):
		labels = tf.concat(rest_of_y, group_y)

	with tf.name_scope("model"):
		weights = tf.Variable(tf.random_normal([num_features, 1]))
		bias = tf.Variable(0.0)
		output_rest = tf.matmul(rest_of_x, weights) + bias
		output_group = tf.matmul(group_x, weight) + bias
		output = tf.concat(output_rest, output_group)
		weight_summary = tf.summary.histogram("weightSummary", weights)
		bias_summary = tf.summary.histogram("bias_summary", bias)

	with tf.name_scope("lossFunction"):
		with tf.name_scope("placeholder"):
			lam_value = tf.constant(lam)

		with tf.name_scope("expected_loss_sub"):
			zero = tf.constant(0.0)
			group_loss = tf.losses.mean_squared_error(group_y, output_group)

		squared_error = tf.losses.mean_squared_error(labels, output)

		with tf.name_scope("regularizer"):#need to scale these by the size of the other
			exp = tf.exp(group_loss - squared_error)
			reg = tf.log(tf.constant(1.0)+exp)#fix to have sub loss not be a fixed input so gradients can pass
		loss = squared_error + lam_value * reg
		loss_summary = tf.summary.scalar("lossSummary", loss)

	with tf.name_scope("optimizer"):
		opt = tf.train.GradientDescentOptimizer(
			learning_rate=learn_rate).minimize(loss)

	saver = tf.train.Saver()
	iteration = 0

	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("graph", sess.graph)

		sess.run(tf.initializers.global_variables())

		for batch in batches:
			for epoch in range(num_epochs):
				sub_x, sub_y = rd.get_sub(batch, groups[epoch % num_groups], column)

				if(len(sub_x) < 1 or len(sub_y) < 1):
					sub_loss = sess.run(zero)
				else:
					sub_loss = sess.run(squared_error, 
						feed_dict={x_features:sub_x, 
						labels:sub_y})


				summary, overall_loss, _ = sess.run([merged, squared_error, opt], 
					feed_dict={x_features:batch["x"], 
					labels:batch["y"], 
					group_loss:sub_loss})

				if(epoch == num_epochs - 1):
					print("summary written")
					writer.add_summary(summary, iteration)
					iteration = iteration + 1

model()
