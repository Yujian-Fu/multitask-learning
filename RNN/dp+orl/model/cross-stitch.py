import tensorflow as tf
import numpy as np 
from tensorflow.python.ops import variable_scope as vs 




class dp4orl(object):
    
    def __init__(self, config, embeddings):
        self.sentences = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentences')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')  # this is not onehot!        
        self.sentence_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_lens')
        self.ds = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ds_ids')
        self.ds_len = tf.placeholder(dtype=tf.int32, shape=[None], name='ds_len')  # length of every PADDED sequence in the batch
        self.ctx = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ctx')
        self.ctx_len = tf.placeholder(dtype=tf.int32, shape=[None], name='ctx_len')
        self.m = tf.placeholder(dtype=tf.int32, shape=[None, None], name='m')
        self.keep_rate_input = tf.placeholder(dtype=tf.float32, name='keep_rate_input')
        self.keep_rate_output = tf.placeholder(dtype=tf.float32, name='keep_rate_output')
        self.keep_state_rate = tf.placeholder(dtype=tf.float32, name='state_keep_rate')
        self.lr = tf.placeholder(dtype = tf.float32, name = 'learning_rate')


        with tf.variable_scope('embeddings_lookup'):
            flag = False

            if isinstance(embeddings, np.ndarray):
            	self.embeddings = tf.get_variable('pretrained_emb',
            		shape=embeddings.shape,
            		initializer=flag,
            		trainable=flag,
            		dtype=tf.float32)

            	embedded_tokens = tf.nn.embedding_lookup(self.embeddings, self.sentences)
            	embedded_ctx = tf.nn.embedding_lookup(self.embeddings, self.ctx)
            	embedded_ds = tf.nn.embedding_lookup(self.embeddings, self.ds)
            	embedded_tokens = tf.identity(embedded_tokens, name='embedded_token')
            	embedded_ctx = tf.identity(embedded_ctx, name='embedded_ctx')

            embedded_ctx_sum = tf.reduce_sum(embedded_ctx, reduction_indices=-2)
            #all ctx in a sentence added to a [sentence_num, 300]variable
            embedded_ctx_mean = embedded_ctx_sum / tf.cast(tf.expand_dims(self.ctx_len, -1), dtype=tf.float32)
            #divide by the number of valid ctx
            embedded_ctx_mean_copy = tf.expand_dims(embedded_ctx_mean, 1)
            #get the tensor with shape [sentence_num, 1,300]
            pattern = tf.stack([1, tf.shape(self.sentences)[1], 1])
            #get the tensor with shape([1,tf.shape(self.sentences)[1], 1]),shape(self.sentences)[1]is the length of sentence
            embedded_ctx_mean_copy = tf.tile(embedded_ctx_mean_copy, pattern)
            #copy length times

            embedded_ds_sum = tf.reduce_sum(embedded_ds, reduction_indices=-2)
            embedded_ds_mean = embedded_ds_sum / tf.cast(tf.expand_dims(self.ds_len, -1), dtype=tf.float32)
            embedded_ds_mean_copy = tf.expand_dims(embedded_ds_mean, 1)
            pattern = tf.stack([1, tf.shape(self.sentences)[1], 1])
            embedded_ds_mean_copy = tf.tile(embedded_ds_mean_copy, pattern)


            inputs = tf.concat(axis = 2, values = [embedded_tokens,
                                               embedded_ds_mean_copy,
                                               embedded_ctx_mean_copy,
                                               tf.cast(tf.expand_dims(self.m, 2),
                               dtype=tf.float32)])

            '''tf.cast：用于改变某个张量的数据类型'''
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_rate_input)
            inputs = tf.identity(inputs, name='inputs')

        n_classes = [config.n_classes_dp, config.n_classes_orl]

        with vs.variable_scope('shared'):
        	with vs.variable_scope('')



