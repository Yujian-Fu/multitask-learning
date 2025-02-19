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



def load_data():
    # train_X: (60000, 28, 28)60000个28*28的训练数据
    # train_y: (60000,)
    # test_X: (10000, 28, 28)
    # test_y: (10000,)
    train_X = np.load("train_x.npy")
    train_y_1 = np.load("train_y_1.npy")
    train_y_2 = np.load("train_y_2.npy")
    test_X = np.load("test_x.npy")
    test_y_1 = np.load("test_y_1.npy")
    test_y_2 = np.load("test_y_2.npy")
    train_X = train_X[0:40000,:]
    train_y_1 = train_y_1[0:40000,:]
    train_y_2 = train_y_2[0:40000,:]
    test_X = test_X[0:40000,:]
    test_y_1 = test_y_1[0:40000,:]
    test_y_2 = test_y_2[0:40000,:]

    return train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2


def combination(input1, input2):
    combination1 = tf.get_variable("combination1",shape=None, dtype=tf.float32,initializer=0.5)
    combination2 = tf.get_variable("combination2",shape=None, dtype=tf.float32,initializer=0.5)
    output1 = tf.multiply(combination1,input1)
    output2 = tf.multiply(combination2,input2)
    output = output1+output2
    return output

def combination_(input1, input2):
    combination1_ = tf.get_variable("combination1_",shape=None, dtype=tf.float32,initializer=0.5)
    combination2_ = tf.get_variable("combination2_",shape=None, dtype=tf.float32,initializer=0.5)
    output1 = tf.multiply(combination1_,input1)
    output2 = tf.multiply(combination2_,input2)
    output = output1+output2
    return output


def apply_cross_stitch(input1, input2):
    output1 = combination(input1,input2)
    output2 = combination_(input1,input2)

    return output1,output2


def main(args):#输入的数据？
    train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2 = load_data()#加载数据集

    m = train_X.shape[0]#输入的训练数据x的个数
    n_output_1 = test_y_1.shape[1]#所属的类的维度（有多少个类）
    n_output_2 = test_y_2.shape[1]

    lr = args.lr#未知，args并不知道是什么东西
    n_epoch = args.n_epoch
    n_batch_size = args.n_batch_size
    reg_lambda = args.reg_lambda
    keep_prob = args.keep_prob
    cross_stitch_enabled = args.cross_stitch_enabled

    with tf.variable_scope("placeholder"):#让变量有相同的命名，包括tf.get_variable得到的变量
	#即括号里面的所有变量都是叫做placeholder
        X = tf.placeholder(tf.float32, (None, 28, 28, 1), "X")
        y_1 = tf.placeholder(tf.float32, (None, n_output_1), "y_1")
        y_2 = tf.placeholder(tf.float32, (None, n_output_2), "y_2")
        is_training = tf.placeholder(tf.bool, (), "is_training")

    with tf.variable_scope("network"):#构建网络
	#括号里面所有的变量都是叫做network
        with contrib.framework.arg_scope(
                [contrib.layers.fully_connected, slim.layers.conv2d],
                # he initialization
                weights_initializer=contrib.layers.variance_scaling_initializer(),
                # l2 regularization
                weights_regularizer=contrib.layers.l2_regularizer(reg_lambda),
                # BN
                normalizer_fn=contrib.layers.batch_norm,
                normalizer_params={
                    "is_training": is_training,
                    "scale": True,
                    "updates_collections": None
                }
        ):
            # (?, 28, 28, 1) -> (?, 28, 28, 32)
            conv1_1 = slim.layers.conv2d(X, 32, kernel_size=[3, 3], scope="conv1_1")#两层卷积两层pooling层
            conv1_2 = slim.layers.conv2d(X, 32, kernel_size=[3, 3], scope="conv1_2")

            # (?, 28, 28, 32) -> (?, 14, 14, 32)
            pool1_1 = slim.layers.max_pool2d(conv1_1, kernel_size=[2, 2], stride=2, scope="pool_1_1")
            pool1_2 = slim.layers.max_pool2d(conv1_2, kernel_size=[2, 2], stride=2, scope="pool_1_2")

            if cross_stitch_enabled:#表示是否构建cross_stitch单元
                with tf.variable_scope("cross_stitch_1"):
                    stitch_pool1_1, stitch_pool1_2 = apply_cross_stitch(pool1_1, pool1_2)
            else:
                stitch_pool1_1, stitch_pool1_2 = pool1_1, pool1_2

            # (?, 14, 14, 32) -> (?, 14, 14, 64)
            conv2_1 = slim.layers.conv2d(stitch_pool1_1, 64, kernel_size=[3, 3], scope="conv2_1")
            conv2_2 = slim.layers.conv2d(stitch_pool1_2, 64, kernel_size=[3, 3], scope="conv2_2")

            # (?, 14, 14, 64) -> (?, 7, 7, 64)
            pool2_1 = slim.layers.max_pool2d(conv2_1, kernel_size=[2, 2], stride=2, scope="pool_2_1")
            pool2_2 = slim.layers.max_pool2d(conv2_2, kernel_size=[2, 2], stride=2, scope="pool_2_2")

            if cross_stitch_enabled:#第二层和第二层的cross_stitch单元
                with tf.variable_scope("cross_stitch_2"):
                    stitch_pool2_1, stitch_pool2_2 = apply_cross_stitch(pool2_1, pool2_2)
            else:
                stitch_pool2_1, stitch_pool2_2 = pool2_1, pool2_2
            print("input")
            print(pool2_1.shape)

            # (?, 7, 7, 64) -> (?, 3136) -> -> (?, 1024)
            with tf.variable_scope("fc_3_1"):#构建全连接层1
                flatten_1 = contrib.layers.flatten(stitch_pool2_1)
                print("fully")
                print(flatten_1.shape)
                fc_3_1 = contrib.layers.fully_connected(flatten_1, 1024)
            with tf.variable_scope("fc_3_2"):#构建全连接层2
                flatten_2 = contrib.layers.flatten(stitch_pool2_2)
                fc_3_2 = contrib.layers.fully_connected(flatten_2, 1024)

            if cross_stitch_enabled:#全连接层之后再次接一个cross_stitch单元
                with tf.variable_scope("cross_stitch_3"):
                    stitch_fc_3_1, stitch_fc_3_2 = apply_cross_stitch(fc_3_1, fc_3_2)
            else:
                stitch_fc_3_1, stitch_fc_3_2 = fc_3_1, fc_3_2

            dropout_1 = contrib.layers.dropout(stitch_fc_3_1, keep_prob=keep_prob, is_training=is_training,
                                               scope="dropout_1")
            dropout_2 = contrib.layers.dropout(stitch_fc_3_2, keep_prob=keep_prob, is_training=is_training,
                                               scope="dropout_2")

            output_1 = contrib.layers.fully_connected(dropout_1, n_output_1, activation_fn=None, scope="output_1")
            output_2 = contrib.layers.fully_connected(dropout_2, n_output_2, activation_fn=None, scope="output_2")

    with tf.variable_scope("loss"):#构建损失函数，分别为任务1和任务2的损失函数
        loss_base_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1, logits=output_1))
        loss_base_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2, logits=output_2))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = loss_base_1 + loss_base_2 + tf.reduce_sum(reg_losses)

    with tf.variable_scope("evaluation"):
        accuracy_1 = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output_1, axis=-1),
            tf.argmax(y_1, axis=-1)), tf.float32), name="accuracy_1")
        accuracy_2 = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output_2, axis=-1),
            tf.argmax(y_2, axis=-1)), tf.float32), name="accuracy_2")
        accuracy = tf.divide(accuracy_1 + accuracy_2, 2.0, name="accuracy")

    with tf.variable_scope("train"):
        global_step = tf.get_variable("global_step", shape=(), dtype=tf.int32, trainable=False)
        lr = tf.train.exponential_decay(lr,global_step, 100, 0.96, staircase=False,name=None)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total, global_step=global_step)

    with tf.variable_scope("summary"):
        summary_loss_total = tf.summary.scalar("loss_total", loss_total)
        summary_accuracy_test = tf.summary.scalar("accuracy_test", accuracy)
        summary_accuracy_train = tf.summary.scalar("accuracy_train", accuracy)

    # standardization
    train_X_reshaped = train_X.reshape([train_X.shape[0], -1])
    train_X_means = np.mean(train_X_reshaped, axis=0, keepdims=True)
    train_X_stds = np.std(train_X_reshaped, axis=0, keepdims=True)

    def standardization(x):
        x_reshaped = x.reshape([x.shape[0], -1])
        result = (x_reshaped - train_X_means) / (train_X_stds + 1e-9)
        return result.reshape(x.shape)

    normalized_test_X = standardization(test_X)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess, tf.summary.FileWriter(
            "./tf_logs/cross-stitch-2X2/result", #+ str(datetime.now().timestamp()),
            graph=tf.get_default_graph()) as f:

        sess.run(tf.global_variables_initializer())

        # similar logic as mnist's next_batch()
        epoch = 0
        index_in_epoch = 0
        while epoch < n_epoch:
            for _ in range(m // n_batch_size + 1):
                start = index_in_epoch
                if start + n_batch_size > m:
                    epoch += 1
                    n_rest_data = m - start
                    train_X_batch_rest = train_X[start:m]
                    train_y_batch_rest_1 = train_y_1[start:m]
                    train_y_batch_rest_2 = train_y_2[start:m]
                    # Shuffle train data
                    perm = np.arange(m)
                    np.random.shuffle(perm)
                    train_X = train_X[perm]
                    train_y_1 = train_y_1[perm]
                    train_y_2 = train_y_2[perm]
                    # Start next epoch
                    start = 0
                    index_in_epoch = n_batch_size - n_rest_data
                    end = index_in_epoch
                    train_X_batch_new = train_X[start:end]
                    train_y_batch_new_1 = train_y_1[start:end]
                    train_y_batch_new_2 = train_y_2[start:end]
                    # concatenate
                    train_X_batch = np.concatenate((train_X_batch_rest, train_X_batch_new), axis=0)
                    train_y_batch_1 = np.concatenate((train_y_batch_rest_1, train_y_batch_new_1), axis=0)
                    train_y_batch_2 = np.concatenate((train_y_batch_rest_2, train_y_batch_new_2), axis=0)
                else:
                    index_in_epoch += n_batch_size
                    end = index_in_epoch
                    train_X_batch = train_X[start:end]
                    train_y_batch_1 = train_y_1[start:end]
                    train_y_batch_2 = train_y_2[start:end]

                _, global_step_value, loss_total_value, summary_loss_total_value = \
                    sess.run([train_op, global_step, loss_total, summary_loss_total],
                             feed_dict={X: standardization(train_X_batch),
                                        y_1: train_y_batch_1,
                                        y_2: train_y_batch_2,
                                        is_training: True})

                if global_step_value % 100 == 0:
                    accuracy_train_value, summary_accuracy_train_value = \
                        sess.run([accuracy, summary_accuracy_train],
                                 feed_dict={X: standardization(train_X_batch),
                                            y_1: train_y_batch_1,
                                            y_2: train_y_batch_2,
                                            is_training: False})
                    accuracy_test_value, summary_accuracy_test_value = \
                        sess.run([accuracy, summary_accuracy_test],
                                 feed_dict={X: normalized_test_X,
                                            y_1: test_y_1,
                                            y_2: test_y_2,
                                            is_training: False})

                    print(global_step_value, epoch, loss_total_value, accuracy_train_value, accuracy_test_value)
                    # cross_stitches = tf.get_collection("cross_stitches")
                    # print(cross_stitches[0].eval(sess))

                    f.add_summary(summary_loss_total_value, global_step=global_step_value)
                    f.add_summary(summary_accuracy_train_value, global_step=global_step_value)
                    f.add_summary(summary_accuracy_test_value, global_step=global_step_value)
                    f.add_graph(sess.graph)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=40)
    parser.add_argument("--n_batch_size", type=int, help="mini batch size", default=128)
    parser.add_argument("--reg_lambda", type=float, help="L2 regularization lambda", default=1e-5)
    parser.add_argument("--keep_prob", type=float, help="Dropout keep probability", default=0.8)
    parser.add_argument("--cross_stitch_enabled", type=bool, help="Use Cross Stitch or not", default=True)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
