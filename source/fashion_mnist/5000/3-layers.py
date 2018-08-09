# -*- coding: utf-8 -*- 
#
# fashion_mnist labels:
# Label	Description
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot
#
# new labels:
# 0 shoe: 5,7,9
# 1 girl: 3,6,8
# 2 other: 0,1,2,4

import sys
import argparse
import numpy as np 
from datetime import datetime
import tensorflow as tf 
import tensorflow.contrib as contrib
import tensorflow.contrib.slim as slim
import keras.datasets.fashion_mnist as fashion_mnist
from keras.utils import to_categorical


def load_data():
	#train_x: (60000,28,28)
	#train_y: (60000,)
	#test_x: (10000,28,28)
	#test_y: (10000,)
	(train_x,train_y_1),(test_x,test_y_1) = fashion_mnist.load_data()
	perm = np.arange(60000)
	np.random.shuffle(perm)
	train_x = train_x[perm]
	train_y_1 = train_y_1[perm]
	train_x = train_x[0:5000,:,:]
	train_y_1 = train_y_1[0:5000]
	n_class_1 = 10
	#map old labels to new ones
	train_y_2 = list(0 if y in [5,7,9] else 1 if y in [3,6,8] else 2 for y in train_y_1)
	test_y_2 = list(0 if y in [5,7,9] else 1 if y in [3,6,8] else 2 for y in test_y_1)
	n_class_2 = 3

	train_x = np.expand_dims(train_x, axis=3)#28X28X1
	#what is expand for?
	test_x = np.expand_dims(test_x,axis=3)
	train_y_1 = to_categorical(train_y_1, n_class_1)
	test_y_1 = to_categorical(test_y_1, n_class_1)
	train_y_2 = to_categorical(train_y_2, n_class_2)
	test_y_2 = to_categorical(test_y_2, n_class_2)

	return train_x, train_y_1, train_y_2, test_x, test_y_1, test_y_2


def combination_2(input1,input2):
	combination1 = tf.get_variable("combination1",shape=None, dtype=tf.float32,initializer=0.5)

	output1 = tf.multiply(combination1,input1)

	combination2 = tf.get_variable("combination2",shape=None, dtype=tf.float32,initializer=0.5)
	output2 = tf.multiply(combination2,input2)

	output = output1 + output2

	return output 

def combination_3(input1, input2, input3):
	combination1 = tf.get_variable("combination1", shape=None, dtype=tf.float32, initializer=0.33)
	output1 = tf.multiply(combination1,input1)

	combination2 = tf.get_variable("combination2",shape=None, dtype=tf.float32,initializer=0.33)
	output2 = tf.multiply(combination2,input2)

	combination3 = tf.get_variable("combination3",shape=None, dtype=tf.float32, initializer=0.33)
	output3 = tf.multiply(combination3,input3)

	output = output1 + output2 + output3

	return output



def main(args):#? where is args from?
	train_x, train_y_1, train_y_2, test_x, test_y_1, test_y_2 = load_data()

	m = train_x.shape[0]#number of samlpes
	n_output_1 = test_y_1.shape[1]#number of outputs
	n_output_2 = test_y_2.shape[1]

	lr = args.lr
	n_epoch = args.n_epoch
	n_batch_size = args.n_batch_size
	reg_lambda = args.reg_lambda
	keep_prob = args.keep_prob
	#sharing_layer_enabled = args.sharing_layer_enabled
	#lr--learning rate
	#n_epoch--number of epoch
	#n_batch_size--mini batch size
	#reg_lamba--L2 regularization lambda
	#keep_prob--dropout keep probability
	#sharing_layer_enabled--add a cross stitch or not
	
	with tf.variable_scope("placeholder"):
		x = tf.placeholder(tf.float32,(None,28,28,1),"x")
		y_1 = tf.placeholder(tf.float32,(None,n_output_1),"y_1")
		y_2 = tf.placeholder(tf.float32,(None,n_output_2),"y_2")
		is_training = tf.placeholder(tf.bool,(),"is_training")

	with tf.variable_scope("network"):
		with contrib.framework.arg_scope(
			[contrib.layers.fully_connected,slim.layers.conv2d],
			weights_initializer=contrib.layers.variance_scaling_initializer(),
			weights_regularizer=contrib.layers.l2_regularizer(reg_lambda),
			normalizer_fn=contrib.layers.batch_norm,
			normalizer_params={
				"is_training": is_training,
				"scale": True,
				"updates_collections": None
			}
		):

			#(?,28,28,1)->(?,28,28,32)
			conv1_1 = slim.layers.conv2d(x,32,kernel_size=[3,3],scope="conv1_1")
			conv1_2 = slim.layers.conv2d(x,32,kernel_size=[3,3],scope="conv1_2")

			#(?,28,28,32)->(?,14,14,32)
			pool1_1 = slim.layers.max_pool2d(conv1_1,kernel_size=[2,2],stride=2,scope="pool1_1")
			pool1_2 = slim.layers.max_pool2d(conv1_2,kernel_size=[2,2],stride=2,scope="pool1_2")

			#(?,14,14,32)->(?,14,14,64)
			conv2_1 = slim.layers.conv2d(pool1_1,64,kernel_size=[3,3],scope="conv2_1")
			conv2_2 = slim.layers.conv2d(pool1_2,64,kernel_size=[3,3],scope="conv2_2")

			#(?,14,14,64)->(?,7,7,64)
			pool2_1 = slim.layers.max_pool2d(conv2_1,kernel_size=[2,2],stride=2,scope="pool2_1")
			pool2_2 = slim.layers.max_pool2d(conv2_2,kernel_size=[2,2],stride=2,scope="pool2_2")

			with tf.variable_scope("combination1"):
				com_result1 = combination_2(pool1_1,pool1_2)
				conv2_3 = slim.layers.conv2d(com_result1,64,kernel_size=[3,3],scope="conv2_3")
				pool2_3 = slim.layers.max_pool2d(conv2_3,kernel_size=[2,2],stride=2,scope="pool2_3")



			#(?,7,7,64)->(?,7,7,128)
			conv3_1 = slim.layers.conv2d(pool2_1, 128, kernel_size=[3,3], scope="conv3_1")
			conv3_2 = slim.layers.conv2d(pool2_2, 128, kernel_size=[3,3], scope="conv3_2")

			#(?,7,7,64)->(?,4,4,128)
			pool3_1 = slim.layers.max_pool2d(conv3_1, kernel_size=[2,2], stride=2, scope="pool3_1")
			pool3_2 = slim.layers.max_pool2d(conv3_2, kernel_size=[2,2], stride=2, scope="pool3_2")


			with tf.variable_scope("combanation2"):
				com_result2 = combination_3(pool2_1,pool2_2,pool2_3)
				conv3_3 = slim.layers.conv2d(com_result2, 128, kernel_size=[3,3], scope="conv3_3")
				pool3_3 = slim.layers.max_pool2d(conv3_3, kernel_size=[2,2], stride=2, scope="pool3_3")



			with tf.variable_scope("combination2_1"):
				com_result2_1 = combination_2(pool3_1,pool3_3)


			with tf.variable_scope("combination2_2"):
				com_result2_2 = combination_2(pool3_2,pool3_3)




			with tf.variable_scope("fc_3_1"):
				flatten_1 = contrib.layers.flatten(com_result2_1)
				fc_3_1 = contrib.layers.fully_connected(flatten_1,1024)
			with tf.variable_scope("fc_3_2"):
				flatten_2 = contrib.layers.flatten(com_result2_2)
				fc_3_2 = contrib.layers.fully_connected(flatten_2,1024)

			dropout_1 = contrib.layers.dropout(fc_3_1,keep_prob=keep_prob,is_training=is_training,
									scope="dropout_1")
			dropout_2 = contrib.layers.dropout(fc_3_2,keep_prob=keep_prob,is_training=is_training,
									scope="dropout_2")

			output_1 = contrib.layers.fully_connected(dropout_1,n_output_1,activation_fn=None,scope="output_1")
			output_2 = contrib.layers.fully_connected(dropout_2,n_output_2,activation_fn=None,scope="output_2")

	with tf.variable_scope("loss"):
		loss_base_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1,logits=output_1))
		loss_base_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2,logits=output_2))
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss_total = loss_base_1 + loss_base_2 + tf.reduce_sum(reg_losses)

	with tf.variable_scope("evaluation"):
		accuracy_1 = tf.reduce_mean(tf.cast(tf.equal(
			tf.argmax(output_1,axis=-1),
			tf.argmax(y_1,axis=-1)),tf.float32),name="accuracy_1")
		accuracy_2 = tf.reduce_mean(tf.cast(tf.equal(
			tf.argmax(output_2,axis=-1),
			tf.argmax(y_2,axis=-1)),tf.float32),name="accuracy_2")
		accuracy = tf.divide(accuracy_1 + accuracy_2,2.0,name="accuracy")

	with tf.variable_scope("train"):
		global_step = tf.get_variable("global_step", shape=(),dtype=tf.int32,trainable=False)
		train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total,global_step=global_step)

	with tf.variable_scope("summary"):
		summary_loss_total = tf.summary.scalar("loss_total",loss_total)
		summary_accuracy_test = tf.summary.scalar("sccuracy_test",accuracy)
		summary_accuracy_train = tf.summary.scalar("accuracy_train",accuracy)

	#standardization
	train_x_reshaped = train_x.reshape([train_x.shape[0],-1])
	train_x_means = np.mean(train_x_reshaped,axis=0,keepdims=True)
	train_x_stds = np.std(train_x_reshaped,axis=0,keepdims=True)

	def standardization(x):
		x_reshaped = x.reshape([x.shape[0], -1])
		result = (x_reshaped - train_x_means)/(train_x_stds + 1e-9)
		return result.reshape(x.shape)

	normalized_test_x = standardization(test_x)

	with tf.Session() as sess, tf.summary.FileWriter(
			"./tf_logs/3-layers/result", #+str(datetime.now().timestamp()),
			graph=tf.get_default_graph()) as f:
		sess.run(tf.global_variables_initializer())

		# similiar logic as mnist's next_batch()
		epoch = 0
		index_in_epoch = 0
		while epoch < n_epoch:
			for _ in range(m // n_batch_size + 1):
				start = index_in_epoch
				if start + n_batch_size > m:
					epoch += 1
					n_rest_data = m - start
					train_x_batch_rest = train_x[start:m]
					train_y_batch_rest_1 = train_y_1[start:m]
					train_y_batch_rest_2 = train_y_2[start:m]

					#shuffle train data
					perm = np.arange(m)
					np.random.shuffle(perm)
					train_x = train_x[perm]
					train_y_1 = train_y_1[perm]
					train_y_2 = train_y_2[perm]

					#start next epoch
					start = 0
					index_in_epoch = n_batch_size - n_rest_data
					end = index_in_epoch
					train_x_batch_new = train_x[start:end]
					train_y_batch_new_1 = train_y_1[start:end]
					train_y_batch_new_2 = train_y_2[start:end]

					#concatentate
					train_x_batch = np.concatenate((train_x_batch_rest,train_x_batch_new),axis=0)
					train_y_batch_1 = np.concatenate((train_y_batch_rest_1,train_y_batch_new_1),axis=0)
					train_y_batch_2 = np.concatenate((train_y_batch_rest_2,train_y_batch_new_2),axis=0)
				else:
					index_in_epoch += n_batch_size
					end = index_in_epoch
					train_x_batch = train_x[start:end]
					train_y_batch_1 = train_y_1[start:end]
					train_y_batch_2 = train_y_2[start:end]

				_, global_step_value,loss_total_value, summary_loss_total_value = \
					sess.run([train_op, global_step, loss_total, summary_loss_total],
						feed_dict={x:standardization(train_x_batch),
									y_1: train_y_batch_1,
									y_2: train_y_batch_2,
									is_training: True})

				if global_step_value % 100 == 0:
					accuracy_train_value,summary_accuracy_train_value = \
						sess.run([accuracy,summary_accuracy_train],
							feed_dict={x:standardization(train_x_batch),
										y_1: train_y_batch_1,
										y_2: train_y_batch_2,
										is_training: False})
					accuracy_test_value, summary_accuracy_test_value = \
						sess.run([accuracy,summary_accuracy_test],
							feed_dict={x:normalized_test_x,
										y_1: test_y_1,
										y_2: test_y_2,
										is_training: False})

					print(global_step_value,epoch,loss_total_value,accuracy_train_value,accuracy_test_value)

					f.add_summary(summary_loss_total_value, global_step=global_step_value)
					f.add_summary(summary_accuracy_train_value,global_step=global_step_value)
					f.add_summary(summary_accuracy_test_value,global_step=global_step_value)
					f.add_graph(sess.graph)


def parse_args(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument("--lr",type=float, help="learning rate", default=0.001)
	parser.add_argument("--n_epoch", type=int, help="number of epoch", default=120)
	parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=128)
	parser.add_argument("--reg_lambda", type=float, help="L2 regularization lambda",default=1e-5)
	parser.add_argument("--keep_prob",type=float, help="Dropout keep probability", default=0.8)
	parser.add_argument("--sharing_layer_enabled", type=bool, help="Use sharing layeror not", default=True)
	
	return parser.parse_args(argv)

if __name__ == "__main__":
	main(parse_args(sys.argv[1:]))





















	
	
	
	




