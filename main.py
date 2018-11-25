import numpy as np
import tensorflow as tf
import read_data as rd

file="clean_dataset.txt"
trial = "1"

def model():
	tf.summary.FileWriterCache.clear()
	learn_rate = 0.0001#need to determine an optimal learning rate
	batch_size = 25#need to determine a good batch size based on total dataset size
	num_epochs = 10#need to determine a good epoch amount
	iterations = 3
	num_groups = 2
	lam = 0.0#should vary this in order to determine impact on accuracy as well as fairness
	groups = [0, 1]# should have values of group
	column = 2#will need to obtain from dataset
	#procedure for choosing optimal lam for a dataset
	features, labels,test_features, test_labels = rd.read_data(filename=file)
	#need to shuffle the data. This is causing a problem with the second half overriding the first half
	#hence causing problems
	batches = rd.get_batches(features, labels, batch_size)
	#shuffle data then take 80% for training 20% for validation

	num_features = len(features[0])#determine num_features based on 


	with tf.name_scope("features"):
		with tf.name_scope("subGroup"):
			len_group = tf.placeholder(tf.float32, shape=(), name="group_len")
			group_x = tf.placeholder(tf.float32, shape=(None, num_features))

		with tf.name_scope("remainingData"):
			len_rest = tf.placeholder(tf.float32, shape=(), name="rest_len")
			rest_of_x = tf.placeholder(tf.float32, shape=(None, num_features))

	with tf.name_scope("labels"):
		with tf.name_scope("subGroup"):
			group_y = tf.placeholder(tf.float32, shape=(None, 1))

		with tf.name_scope("remainingData"):
			rest_of_y = tf.placeholder(tf.float32, shape=(None, 1))

		labels = tf.concat([rest_of_y, group_y], 0)

	with tf.name_scope("model"):
		weights = tf.Variable(tf.random_normal([num_features, 1]), name="weights")
		bias = tf.Variable(0.0, name="bias")

		with tf.name_scope("remainingData"):
			output_rest = tf.matmul(rest_of_x, weights) + bias

		with tf.name_scope("subGroup"):
			output_group = tf.matmul(group_x, weights) + bias

		output = tf.concat([output_rest, output_group], 0)

		with tf.name_scope("summaries"):
			weight_summary = tf.summary.histogram("weightSummary", weights)
			bias_summary = tf.summary.histogram("bias_summary", bias)

	with tf.name_scope("lossFunction"):
		with tf.name_scope("placeholder"):
			lam_value = tf.constant(lam)

		with tf.name_scope("expected_loss_sub"):
			zeros = tf.zeros((1,num_features), dtype=tf.float32)
			group_loss = tf.losses.mean_squared_error(group_y, output_group)

		squared_error = tf.losses.mean_squared_error(labels, output)

		with tf.name_scope("regularizer"):#need to scale these by the size of the other
			exp = len_rest * group_loss - len_group * squared_error
			reg = tf.nn.softplus(exp)
		
		loss = squared_error + lam_value * reg

		with tf.name_scope("summaries"):
			loss_summary = tf.summary.scalar("lossSummary", loss)

	with tf.name_scope("optimizer"):
		optim = tf.train.GradientDescentOptimizer(
			learning_rate=learn_rate)
		grad = optim.compute_gradients(loss)
		opt = optim.minimize(loss)

	saver = tf.train.Saver()
	iteration = 0

	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("graph/trial" + trial, sess.graph)

		sess.run(tf.initializers.global_variables())
		for rnd in range(iterations):
			for batch in batches:
				for epoch in range(num_epochs):
					sub_x, sub_y, rest_x, rest_y = rd.get_groups(batch, groups[epoch % num_groups], column)

					if len(sub_y) < 1:
						sub_y = np.reshape([sess.run(bias)], (1, 1))
						sub_x = sess.run(zeros)
					if len(rest_y) < 1:
						rest_y = np.reshape([sess.run(bias)], (1, 1))
						rest_x = sess.run(zeros)

					grads, summary, overall_loss, _ = sess.run([grad, merged, squared_error, opt], 
						feed_dict={group_x:sub_x,
						group_y: sub_y,
						rest_of_x: rest_x,
						rest_of_y: rest_y,
						len_group: len(sub_y),
						len_rest: len(rest_y)})

					if(epoch % 5 == 0):
						#print("summary written")
						writer.add_summary(summary, global_step=iteration)
						out_val = sess.run(output_group, 
									feed_dict={group_x:test_features})

						num_correct = 0
						out_val = np.reshape(out_val, (1, -1))

						for i in range(len(test_labels)):
							if(out_val[0][i] > 0 and test_labels[i] == 1):
								num_correct += 1
							elif(out_val[0][i] <= 0 and test_labels[i] == -1):
								num_correct += 1

						print(sess.run(
							tf.losses.mean_squared_error(
								tf.reshape(out_val, shape=(1, 750)), tf.reshape(test_labels, shape=(1, 750)))))

						print("Total Correct:")
						print(num_correct)
						print("Out of:")
						print(len(test_labels))
						iteration = iteration + 1


model()
