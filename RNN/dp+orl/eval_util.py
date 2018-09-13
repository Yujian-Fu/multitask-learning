import numpy as np
import tensorflow as tf
from operator import add
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path


def eval_orl(batches, sess, seq_model, task_id):
    print("evaluating")
    intersect_binary = [0] * 3
    intersect_proportional = [0] * 3
    num_gold = [0] * 3
    num_pred = [0] * 3
    for batch in batches:
        pred_train, true_train = test_step(sess, seq_model, batch, task_id)
        intersect_binary_temp, intersect_proportional_temp, num_pred_temp, num_gold_temp = f_measure_orl(pred_train, true_train)
        intersect_binary = list(map(add, intersect_binary, intersect_binary_temp))
        intersect_proportional = list(map(add, intersect_proportional, intersect_proportional_temp))
        num_gold = list(map(add, num_gold, num_gold_temp))
        num_pred = list(map(add, num_pred, num_pred_temp))
        
    fscores_dev = f_measure_final(intersect_binary, intersect_proportional, num_pred, num_gold)
    macro_binary_fscore_dev = fscores_dev[0]
    macro_proportional_fscore_dev = fscores_dev[1]
    return macro_binary_fscore_dev, macro_proportional_fscore_dev

def test_step(sess, seq_model, data, task_id):
    sentences, labels, ds, ds_len, ctx, ctx_len, m, sentence_lens = zip(*data)

    assert len(sentences) > 0
    assert len(labels) > 0

    #print("an sample of data for test:", sentences[0])
    feed_dict = {
        seq_model.sentences: list(sentences),  # batch_data_padded_x,
        seq_model.labels: list(labels),  # batch_data_padded_y,
        seq_model.sentence_lens: list(sentence_lens),  # batch_data_seqlens
        seq_model.ds: list(ds),
        seq_model.ds_len: list(ds_len),
        seq_model.ctx: list(ctx),
        seq_model.ctx_len: list(ctx_len),
        seq_model.m: list(m),
        seq_model.keep_rate_input: 1.0,
        seq_model.keep_rate_output: 1.0,
        seq_model.keep_state_rate: 1.0
    }
    transition_params_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/transition_params').outputs[0]
    unary_scores_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/unary_scores').outputs[0]

    tf_unary_scores, tf_transition_params = sess.run([unary_scores_op, transition_params_op], feed_dict)

    predictions = []
    gold = []

    for i in range(len(sentences)):
        #print("in every prediction")
        length = sentence_lens[i]
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores[i, :length, :], tf_transition_params)
        #print("the unary and the target in this test batch and its shape is",tf_transition_params)
        #print("the viterbi_sequence is", viterbi_sequence)
        predictions.append(viterbi_sequence)
        gold.append(labels[i][:length])
        #the prediction is always 0?
    return predictions, gold

def f_measure_final(intersect_binary, intersect_proportional, num_pred, num_gold):
    precision_binary = []
    recall_binary = []
    fscore_binary = []

    precision_proportional = []
    recall_proportional = []
    fscore_proportional = []

    for t, type in enumerate(['d', 'h', 't']):
        try:
            print("intersect_binary and num_gold", intersect_binary[t], float(num_gold[t]))
            recall_binary.append(intersect_binary[t] / float(num_gold[t]))
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            recall_binary.append(0.0)

        try:
            recall_proportional.append(intersect_proportional[t] / float(num_gold[t]))
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            recall_proportional.append(0.0)

        try:
            precision_binary.append(intersect_binary[t] / float(num_pred[t]))
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            precision_binary.append(0.0)

        try:
            precision_proportional.append(intersect_proportional[t] / float(num_pred[t]))
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            precision_proportional.append(0.0)

        try:
            fscore_binary.append(
                2.0 * precision_binary[t] * recall_binary[t] / float(precision_binary[t] + recall_binary[t]))
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            fscore_binary.append(0.0)

        try:
            fscore_proportional.append(2.0 * precision_proportional[t] * recall_proportional[t] /
                                                            float(precision_proportional[t] + recall_proportional[t]))
        except ZeroDivisionError:
            fscore_proportional.append(0.0)
    return [np.nan_to_num(fscore_binary), np.nan_to_num(fscore_proportional)]

def f_measure_orl(prediction, target):
    #print("evaluation in each epoch")
    beggining_label = {'d': 1, 'h': 3, 't': 5}

    intersect_binary = [0]*3
    intersect_proportional = [0]*3
    num_gold = [0]*3
    num_pred = [0]*3
    for t, type in enumerate(['d', 'h', 't']):
        gold = []
        for i in range(len(target)):
            previous = -1
            for j in range(len(target[i])):
                entity = []
                b = beggining_label[type]
                if (target[i][j] == b) or (target[i][j] == b+1 and previous not in [b, b+1]):
                    entity.append((i, j))
                    flag = 0
                    for k in range(j+1, len(target[i])):
                        if (target[i][k] == b+1) and (flag == 0):
                            entity.append((i, k))
                        else:
                            flag = 1
                    if entity:
                        gold.append(entity)
                    previous = target[i][j]

        predicted = []
        for i in range(len(prediction)):
            previous = -1
            for j in range(len(prediction[i])):
                entity = []
                b = beggining_label[type]
                if (prediction[i][j] == b) or (prediction[i][j] == b+1 and previous not in [b, b+1]):
                    entity.append((i, j))
                    flag = 0
                    for k in range(j + 1, len(prediction[i])):
                        if (prediction[i][k] == b + 1) and (flag == 0):
                            entity.append((i, k))
                        else:
                            flag = 1
                    if entity:
                        predicted.append(entity)
                    previous = prediction[i][j]

        intersect_binary_temp = 0.0
        intersect_proportional_temp = 0.0

        for entity_pred in predicted:
            flag = 0
            for entity_gold in gold:
                if (len(list(set(entity_pred) & set(entity_gold))) >= 1) and (flag == 0):
                    print("successful predicted")                    
                    intersect_binary_temp += 1
                    intersect_proportional_temp += len(list(set(entity_pred) & set(entity_gold))) / float(len(entity_gold))
                    flag = 1
        intersect_binary[t] = intersect_binary_temp
        intersect_proportional[t] = intersect_proportional_temp
        num_gold[t] = len(gold)
        num_pred[t] = len(predicted)
    
    #print("the result in this batch:",intersect_binary,intersect_proportional)
    return intersect_binary, intersect_proportional, num_pred, num_gold

def plot_training_curve(fig_path, num_iter, flist):
    plt.figure(dpi=400)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
    steps = range(1, num_iter + 1)
    plt.plot(steps, flist[0], linewidth=1, color='#6699ff', linestyle='-', marker='o',
             markeredgecolor='black',
             markeredgewidth=0.5, label='train')
    plt.plot(steps, flist[1], linewidth=3, color='#ff4d4d', linestyle='-', marker='D',
             markeredgecolor='black',
             markeredgewidth=0.5, label='test')
    plt.plot(steps, flist[2], linewidth=2, color='#ffcc66', linestyle='-', marker='s',
             markeredgecolor='black',
             markeredgewidth=0.5, label='dev')
    plt.xlabel('epochs')
    plt.ylabel('binary f1')
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)

