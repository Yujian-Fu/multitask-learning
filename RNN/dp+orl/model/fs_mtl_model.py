import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np

class dp4orl(object):
    def __init__(self,config,embeddings):



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
        '''构建输入参数的集合'''


        with tf.variable_scope('embeddings_lookup'):
            flag = False

            if isinstance(embeddings, np.ndarray):
                self.embeddings = tf.get_variable('pretrained_emb',
                                                   shape=embeddings.shape,
                                                   initializer=tf.constant_initializer(embeddings),
                                                   trainable=flag,
                                                   dtype=tf.float32)

            '''根据索引搜索张量中的索引对应的值'''
              


                
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

            stacked_fw = []
            stacked_bw = []
            for _ in range(config.n_layers_shared):
                with vs.variable_scope('forward'):
                    cell_fw = tf.contrib.rnn.LSTMCell(num_units=config.hidden_size,
                                                      state_is_tuple=True,
                                                      initializer=tf.orthogonal_initializer(seed=config.seed))
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                        variational_recurrent=True,
                                                        state_keep_prob=self.keep_state_rate,
                                                        output_keep_prob=self.keep_rate_output,
                                                        dtype=tf.float32)
                    stacked_fw.append(cell_fw)

                with vs.variable_scope('backward'):
                    cell_bw = tf.contrib.rnn.LSTMCell(num_units=config.hidden_size,
                                                      state_is_tuple=True,
                                                      initializer=tf.orthogonal_initializer(seed=config.seed))
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                        variational_recurrent=True,
                                                        state_keep_prob=self.keep_state_rate,
                                                        output_keep_prob=self.keep_rate_output,
                                                        dtype=tf.float32)
                    stacked_bw.append(cell_bw)

            multicell_fw = tf.contrib.rnn.MultiRNNCell(stacked_fw, state_is_tuple=True)
            multicell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw, state_is_tuple=True)

            self.outputs_shared_tuple, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell_fw,
                                                                      cell_bw=multicell_bw,
                                                                      dtype=tf.float32,
                                                                      sequence_length=self.sentence_lens,
                                                                      inputs=inputs)
            outputs_fw, outputs_bw = self.outputs_shared_tuple
            self.outputs_shared = tf.concat(axis=2, values=[outputs_fw, outputs_bw], name='self.outputs_shared')
            self.outputs_shared = tf.identity(self.outputs_shared, name='outputs')


        for task_id in range(2):
            with tf.variable_scope('task'+str(task_id)):
                outputs_flat = tf.reshape(self.outputs_shared, [-1, 2*config.hidden_size], name='outputs_flat')

                out_weights = tf.get_variable(name='out_w', shape= [2*config.hidden_size, n_classes[task_id]],
                                               dtype=tf.float32,
                                               trainable=True)
                out_biases = tf.get_variable( name='out_b',
                                          shape=[n_classes[task_id]],
                                          initializer=tf.constant_initializer(value=0),
                                          dtype=tf.float32,
                                          #regularizer=regularizer,
                                          trainable=True)
                matricized_unary_scores = tf.matmul(outputs_flat, tf.nn.dropout(out_weights, keep_prob=self.keep_rate_output)) + out_biases

                self.unary_scores = tf.reshape(matricized_unary_scores, [tf.shape(self.sentences)[0], -1, n_classes[task_id]],
                                               name='unary_scores')

                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores,
                                                                                  self.labels,
                                                                                  self.sentence_lens)
                self.transition_params = tf.identity(transition_params, name='transition_params')
                self.task_losses = tf.reduce_mean(-log_likelihood, name='task_logl_loss')
                tv = tf.trainable_variables()
                self.regularization_cost = config.reg_coef * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv],
                                                                    name='regularization_cost')
                total_loss = tf.add(self.task_losses, self.regularization_cost, name='total_loss')

        with tf.variable_scope('train_step'):
          optimizer = tf.train.AdamOptimizer(self.lr)
          self.train_op = optimizer.minimize(total_loss, name = 'train_op')