import numpy as np
import tensorflow as tf


def _train_ops(learning_rate, grad_clip):
	train_ops = []
	for task_id in range(2):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		loss = tf.get_default_graph().get_tensor_by_name('task'+str(task_id)+'/total_loss:0')
		if grad_clip>0:
			grads, vs     = zip(*optimizer.compute_gradients(loss))
			grads, gnorm  = tf.clip_by_global_norm(grads, grad_clip)
			train_op = optimizer.apply_gradients(zip(grads, vs))
		else:
			train_op = optimizer.minimize(loss)

		train_ops.append(train_op)

	return train_ops


def train_task_step(model, sess, task_id, data, config, learning_rate):

    if task_id == 0:
        sentences = []
        labels = []
        ds = []
        ds_len = []
        ctx = []
        ctx_len = []
        m = []
        sentence_lens = []

        for word_target in data:
            sentences.append(word_target[0])
            labels.append(word_target[1])
            ds.append([53235])
            ds_len.append(1)
            ctx.append([53235])
            ctx_len.append(1)
            m.append([0,0,0])
            sentence_lens.append(3)
            
    if task_id == 1:
        sentences, labels, ds, ds_len, ctx, ctx_len, m, sentence_lens = zip(*data)    	
    


    fd = {model.sentences: list(sentences),
          model.labels: list(labels),
          model.sentence_lens: tuple(sentence_lens),
          model.ds: list(ds),
          model.ds_len: list(ds_len),
          model.ctx: list(ctx),
          model.ctx_len: list(ctx_len),
          model.m: list(m),
          model.keep_rate_input: config.keep_rate_input,
          model.keep_rate_output: config.keep_rate_output,
          model.keep_state_rate: config.keep_state_rate,
          model.lr: learning_rate
          }

    #print('the sample of input data is', sentences)#, labels[0], sentence_lens[0], ds[0], ds_len[0],ctx[0],ctx_len[0])
    input_op = tf.get_default_graph().get_operation_by_name('embeddings_lookup/inputs').outputs[0]
    output_op = tf.get_default_graph().get_operation_by_name('shared/outputs').outputs[0]
    embedded_tokens_op = tf.get_default_graph().get_operation_by_name('embeddings_lookup/embedded_token').outputs[0]
    embedded_ctx_op = tf.get_default_graph().get_operation_by_name('embeddings_lookup/embedded_ctx').outputs[0]

    task_loss_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/task_logl_loss').outputs[0]
    transition_params_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/transition_params').outputs[0]
    unary_scores_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/unary_scores').outputs[0]
    variables_names = [v.name for v in tf.trainable_variables()]

    train_op = tf.get_default_graph().get_operation_by_name('train_step/train_op')

    before = sess.run(tf.trainable_variables())

    embedded_token, embedded_ctx, input_, output, loss, tf_unary_scores, tf_transition_params = sess.run(
                                                            [embedded_tokens_op,
                                                            embedded_ctx_op,
                                                            input_op,
                                                            output_op,
                                                            task_loss_op,
                                                            #reg_cost_op,
                                                            unary_scores_op,
                                                            transition_params_op],
                                                            feed_dict = fd)
    _ = sess.run([train_op], feed_dict = fd)

    after = sess.run(tf.trainable_variables())

    #print('the sample of embedded_token', embedded_token[0][0])
    #print('the samlpe of embedded_ctx', embedded_ctx[0][0])
    #print('the sample of input', input_[0][0])
    #print('the sample of output', output[0][0])

    #print('the tf_transition_prams', tf_transition_params)
    #print('the scores is', tf_unary_scores[0])

    for i, (b, a) in enumerate(zip(before, after)):
        # Make sure something changed.
        if variables_names[i].split('/')[0] == 'shared':
            #assert (b != a).any()
            if (a ==b).all():
                print("share layer not changed!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            if int(variables_names[i].split('/')[0][-1]) == task_id:
                #assert (b != a).any()
                if (a ==b).all():
                    print(task_id,"task layer not changed!!!!!!!!!!!!!!!!!!!!!!!")


    predictions = []
    gold = []

    #print('the length of sentences is', len(list(sentences)))
    for i in range(len(list(sentences))):
        length = sentence_lens[i]
        #print("the score is", tf_unary_scores[i, :length, :])
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores[i, :length, :], tf_transition_params)
        predictions.append(viterbi_sequence)
        gold.append(labels[i][:length])
        #print("the length of sentences is", i)
        
    print('gold and the prediction', gold[-1], predictions[-1])
        
        #return predictions, gold
    print("the loss of train is ", loss)
