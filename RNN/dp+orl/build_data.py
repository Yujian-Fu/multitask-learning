from config import config
import os
from collections import OrderedDict
from collections import defaultdict
import json
import numpy as np
from data_util import *
from class_type import DatasetParser

if __name__ == '__main__':
    #load dp data
    with open('./data/dev.conll') as f: 
        dp_dev_corpus = [[w.split() for w in s.split('\n')] for s in f.read().split('\n\n')]
        print("Loaded dp_dev")

    with open('./data/train.conll') as f:
        dp_train_corpus = [[w.split() for w in s.split('\n')] for s in f.read().split('\n\n')]
        print("Loaded dp_train")

    with open('./data/test.conll') as f:
        dp_test_corpus = [[w.split() for w in s.split('\n')] for s in f.read().split('\n\n')]
        print("Loaded dp_test")


    sentence_dp = []


    for sent in dp_train_corpus:
        sentence_dp.append(sent)

    for sent in dp_dev_corpus:
        sentence_dp.append(sent)

    for sent in dp_test_corpus:
        sentence_dp.append(sent)

    sentence_orl = []
    #load orl data
    for fold in range(4):
        dev_name = 'jsons/new/dev.json'
        orl_dev_corpus = json.load(open(dev_name))
        print("Loaded orl_dev")
        print(type(orl_dev_corpus))

        train_name = 'jsons/new/train_fold_'+str(fold)+'.json'
        orl_train_corpus = json.load(open(train_name))
        print("Loaded orl_train")
        print(type(orl_train_corpus))

        test_name = 'jsons/new/test_fold_'+str(fold)+'.json'
        orl_test_corpus = json.load(open(test_name))
        print("Loaded orl_test")
        print(type(orl_test_corpus))


        for doc_num in range(orl_train_corpus['documents_num']):
           doc = orl_train_corpus['document'+str(doc_num)]
           for sent_num in range(doc['sentences_num']):
              sentence_orl.append(map(lambda x: x.lower(),doc['sentence'+str(sent_num)]['sentence_tokenized']))

        for doc_num in range(orl_dev_corpus['documents_num']):
            doc = orl_dev_corpus['document'+str(doc_num)]
            for sent_num in range(doc['sentences_num']):
                sentence_orl.append(map(lambda x: x.lower(),doc['sentence'+str(sent_num)]['sentence_tokenized']))
 
        for doc_num in range(orl_test_corpus['documents_num']):
            doc = orl_test_corpus['document'+str(doc_num)]
            for sent_num in range(doc['sentences_num']):
                sentence_orl.append(map(lambda x: x.lower(),doc['sentence'+str(sent_num)]['sentence_tokenized']))

    highlight("build_vocab")
    all_words, word2idx, idx2word = build_vocab(sentence_orl, sentence_dp)

    highlight("build embedding")
    embedding_dict, embedding_matrix = build_embedding(word2idx, all_words)

    highlight("save vocab and embedding")
    save_list('all_words.txt', all_words)
    save_list(config.word2idx_name,word2idx)
    save_list(config.embedding_dict_name,embedding_dict)
    np.savetxt("embedding_matrix.txt", embedding_matrix)


def transform_dp_data(sentences, embedding_dict, word2idx):

    train_data = []


    for i, sentence in enumerate(sentences):

        parser = DatasetParser(sentence, embedding_dict)

        #sentence_vector = parser.sentence2vec()
        #transfer every word in the sentence to vector

        #print("type parser is:", type(parser))

        sentence_data = dp_transitions(parser, word2idx)
        #get the transfered input_data and the label


        train_data.append(sentence_data)

    return train_data

def dp_transitions(parser,word2idx):
    ms1l = []
    ms2l = []
    mbl = []
    targets = []
    data = []


    while parser.has_next():
            #whether all words in stack and buffer are processed
        (ms1, ms2), mb = parser.get_stack(), parser.get_buffer()
        #self.__stack[-1][0],self.__stack[-2][0], self.__buffer[0][0]
        '''
        ms1l.append(parser.index2mask(ms1))
        ms2l.append(parser.index2mask(ms2))
        mbl.append(parser.index2mask(mb))
        '''
        #return a ndarray with True in index and all false in others

        tag = parser.next()
        target = transition2onehot(tag)
        targets.append(target)

        data.append([[word2idx[ms1],word2idx[ms2],word2idx[mb]],target])
    
    '''        
    ms1l = np.array(ms1l)
    ms2l = np.array(ms2l)
    mbl = np.array(mbl)
    targets = np.array(targets)
    '''
    return data
    #i , j , k: the length of ms1l, the length of sentence, 300


def transition2onehot(tag):
    
    TRANSITIONS = ['reduce-left', 'shift', 'reduce-right']
    a = tag.split('-')
    a = a[0] + '-' + a[1] if len(a) > 1 else a[0]

    target = [0, 0 ,0]
    target[TRANSITIONS.index(a)] = int(1)
    return target

def pad_orl_data(batch, vocab):
    global PAD
    pad_id = vocab[PAD]
    batch_x, _, _, _, _, _, _ = zip(*batch)
    max_length = max([len(inst) for inst in batch_x])


    _, _, ds_all, _, _, _, _ = zip(*batch)
    max_length_ds = max([len(ds) for ds in ds_all])

    _, _, _, _, ctx_all, _, _ = zip(*batch)
    max_length_ctx = max([len(ctx) for ctx in ctx_all])
    '''取出每一个batch中每一维的标签或者数据的最长的长度'''

    batch_pad = []
    for (x, y, ds, ds_len, ctx, ctx_len, m) in batch:
        '''对于batch中的每一组数据都依次处理'''
        assert len(x) == len(y)
        diff = max_length - len(x)
        '''最大值和这个第一维之间的差值'''
        assert diff >= 0
        z = []
        w = []
        p = []
        for _ in range(diff):
            z.append(pad_id)
            w.append(7)
            p.append(0)

        diff_ds = max_length_ds - len(ds)
        '''最大值和当前的ds的长度的差值'''
        assert diff_ds >= 0
        q = []
        for _ in range(diff_ds):
            q.append(pad_id)

        diff_ctx = max_length_ctx - len(ctx)
        assert diff_ctx >= 0
        r = []
        for _ in range(diff_ctx):
            r.append(pad_id)

        batch_pad.append((list(x+z), y+w, ds+q, ds_len,
                        ctx+r, ctx_len, m+p, len(x)))

        assert len(z) == len(w)

    return batch_pad


def get_word_ids(sent, vocabulary, task):
    global UNK
    word_ids = []
    for _, w in enumerate(sent):
        if task == 'srl':
            w_id = vocabulary[w[0].lower()] if w[0].lower() in vocabulary else vocabulary[UNK]

        else:
            w_id = vocabulary[w.lower()] if w.lower() in vocabulary else vocabulary[UNK]
        word_ids.append(w_id)
    return word_ids


def transform_orl_data(corpus, vocabulary, window_size, mode, exp_setup_id='new', att_link_exists_obligatory='false'):
    global UNK
    global PAD

    vocabulary_inv = [None] * len(vocabulary)
    for w in vocabulary:
        vocabulary_inv[vocabulary[w]] = w
        '''wocab中单词所对应的序号在inv中变成序号对应成单词'''

    labels_all = []
    sentences_all = []
    ctx_all = []
    ctx_len = []
    ds_all = []
    ds_len = []
    m_all = []
    ds_indices_all = []
    sentences_original_all = []
    #all data that return

    # data_file = open('mpqa_files/examples_' + mode + '_' + str(fold) + '.txt', 'w')
    type_counts = defaultdict(int)

    stats_dict = defaultdict(int)
    for doc_num in range(corpus['documents_num']):
        document_name = "document" + str(doc_num)
        doc = corpus[document_name]
        for sent_num in range(doc['sentences_num']):
            sentence_name = "sentence" + str(sent_num)
            sentence = doc[sentence_name]['sentence_tokenized']
            '''循环取出某文件中某句子'''

            annos = []
            for ds_id in doc[sentence_name]['dss_ids']:
                ds_name = "ds" + str(ds_id)
                ds = doc[sentence_name][ds_name]
                # ds_all_num += 1
                stats_dict['ds_all_num'] += 1

                if ds['ds_implicit']:
                    # implicit_ds += 1
                    stats_dict['implicit_ds'] += 1
                    continue


                inferred = True
                for atid in range(ds['att_num']):
                    if ds['att' + str(atid)]['attitudes_inferred'] != 'yes':
                        inferred = False

                if inferred:
                    # inferred_ds += 1
                    stats_dict['inferred_ds'] += 1
                    continue


                if not ds['attitude_link_exists']:
                    # no_att_link += 1
                    stats_dict['no_att_link'] += 1
                    if att_link_exists_obligatory == 'true':
                        continue


                if ds['ds_insubstantial'] != 'none':
                    # insubs_ds += 1
                    stats_dict['insubs_ds'] += 1


                if ds['ds_annotation_uncertain'] == 'somewhat-uncertain':
                    # ds_uncertain[0] += 1
                    stats_dict['ds_annotation_somewhat_uncertain'] += 1

                if ds['ds_annotation_uncertain'] == 'very-uncertain':
                    # ds_uncertain[1] += 1
                    stats_dict['ds_annotation_very_uncertain'] += 1

                ds_entity = ds['ds_tokenized']
                ds_indices = ds['ds_indices']
                assert ds_indices

                if len(set(ds_indices) & set(annos)) > 0:
                    # overlap_count += 1
                    stats_dict['overlap_count'] += 1
                annos.extend(ds_indices)


                holder_indices = []
                holder_unique_ids = []
                for i, (hol1, o) in enumerate(zip(ds['holders_indices'], ds['holder_ds_overlap'])):
                    if not o:
                        not_overlap = True
                        for hol2 in holder_indices:
                            if len(set(hol1) & set(hol2)) > 0:
                                not_overlap = False

                        if not_overlap:
                            holder_indices.append(hol1)
                            holder_unique_ids.append(i)

                for hid in holder_indices:
                    if len(set(hid) & set(annos)) > 0:
                        # overlap_count += 1
                        stats_dict['overlap_count'] += 1
                    annos.extend(hid)

                holders = [hol for i, hol in enumerate(ds['holders_tokenized']) if i in holder_unique_ids]

                for u in ds['holders_uncertain']:
                    if u == 'somewhat-uncertain':
                        # holder_uncertain[0] += 1
                        stats_dict['holder_somewhat_uncertain'] += 1
                    if u == 'vert-uncertain':
                        # holder_uncertain[1] += 1
                        stats_dict['holder_very_uncertain'] += 1

                # no duplicate holders allowed
                assert len([' '.join(hol) for hol in holders]) == len(list(set([' '.join(hol) for hol in holders])))

                targets = []
                target_indices = []
                attitudes = []
                ds_attitudes = {}

                if ds['attitude_link_exists']:
                    for aid in range(ds['att_num']):
                        if ds['att' + str(aid)]['attitudes_inferred'] != 'yes':
                            # we associate only one attitude per attitude type
                            ds_attitudes[ds['att' + str(aid)]['attitudes_types']] = aid


                    if ds_attitudes.keys():
                        atypes = ['sentiment', 'intention', 'agree', 'arguing', 'other-attitude', 'speculation']
                        att_not_found = True
                        att_idx = []
                        for tid, atype in enumerate(atypes):
                            if att_not_found:
                                if tid < 4:  # sentiment-pos, sentiment-neg, intention-pos, etc.
                                    for polarity in ['pos', 'neg']:
                                        atype_full = atype + '-' + polarity
                                        if atype_full in ds_attitudes:
                                            att_idx.append(ds_attitudes[atype_full])
                                            attitudes.append(atype_full)
                                            att_not_found = False
                                            type_counts[atype_full] += 1
                                else:
                                    if atype in ds_attitudes:
                                        att_idx.append(ds_attitudes[atype])
                                        attitudes.append(atype)
                                        att_not_found = False
                                        type_counts[atype] += 1

                        for aid in att_idx:
                            att = ds['att' + str(aid)]

                            # do not allow overlapping targets (can mess up the BIO scheme)
                            targets_ind_temp = []
                            target_unique_ids = []
                            for i, (tar1, o) in enumerate(zip(att['targets_indices'], att['target_ds_overlap'])):
                                if not o:
                                    not_overlap = True
                                    for tar2 in targets_ind_temp:
                                        if len(set(tar1) & set(tar2)) > 0:
                                            not_overlap = False

                                    if not_overlap:
                                        targets_ind_temp.append(tar1)
                                        target_unique_ids.append(i)
                            target_indices.extend(targets_ind_temp)

                            targets_temp = [tar for i, tar in enumerate(att['targets_tokenized']) if
                                            i in target_unique_ids]
                            targets.extend(targets_temp)

                            for u in att['targets_uncertain']:
                                if u == 'somewhat-uncertain':
                                    # target_uncertain[0] += 1
                                    stats_dict['target_somewhat_uncertain'] += 1
                                if u == 'vert-uncertain':
                                    # target_uncertain[1] += 1
                                    stats_dict['target_very_uncertain'] += 1

                        for tid in target_indices:
                            if len(set(tid) & set(annos)) > 0:
                                # overlap_count += 1
                                stats_dict['overlap_count'] += 1
                            annos.extend(tid)

                '''
                Start BIO annotations:
                O = 0
                B_DS = 1
                I_DS = 2
                B_H = 3
                I_H = 4
                B_T = 5
                I_T = 6
                '''
                labels = [0] * len(sentence)
                labels[ds_indices[0]] = 1
                if len(ds_indices) > 1:
                    for idx in ds_indices[1:]:
                        labels[idx] = 2

                for holder in holder_indices:
                    labels[holder[0]] = 3
                    if len(holder) > 1:
                        for idx in holder[1:]:
                            labels[idx] = 4

                for target in target_indices:
                    labels[target[0]] = 5
                    if len(target) > 1:
                        for idx in target[1:]:
                            labels[idx] = 6

                ds_vocab_ids = []
                for w in ds_entity:
                    w_id = vocabulary[w.lower()] if w in vocabulary else vocabulary[UNK]
                    ds_vocab_ids.append(w_id)

                # context of the direct-subjective (input to the model)
                ctx = []
                end = min(len(sentence), ds_indices[len(ds_indices) - 1] + window_size + 1)
                begin = max(0, ds_indices[0] - window_size)

                for k in range(begin, end):
                    w = sentence[k]
                    w_id = vocabulary[w.lower()] if w in vocabulary else vocabulary[UNK]
                    ctx.append(w_id)

                if len(sentence) < ds_indices[len(ds_indices) - 1] + window_size + 1:
                    diff = ds_indices[len(ds_indices) - 1] + window_size + 1 - len(sentence)
                    for _ in range(diff):
                        ctx.append(vocabulary[PAD])

                if ds_indices[0] - window_size < 0:
                    diff = -(ds_indices[0] - window_size)
                    for _ in range(diff):
                        ctx.insert(0, vocabulary[PAD])

                # indicator function: 1 if a word is in the context of the DS, 0 otherwise
                m = []
                for j in range(len(sentence)):
                    if j in range(ds_indices[0] - window_size, ds_indices[len(ds_indices) - 1] + window_size + 1):
                        m.append(1)
                    else:
                        m.append(0)


                if len(sentence) > 150:
                    cut_num = 15
                    sentence_cut = sentence[
                                   max(0, ds_indices[0] - cut_num):min(ds_indices[-1] + cut_num, len(sentence))]
                    labels_cut = labels[max(0, ds_indices[0] - cut_num):min(ds_indices[-1] + cut_num, len(sentence))]
                    m_cut = labels[max(0, ds_indices[0] - cut_num):min(ds_indices[-1] + cut_num, len(sentence))]

                    sentences_original_all.append(' '.join(sentence_cut))
                    m_all.append(m_cut)
                    labels_all.append(labels_cut)
                    sentence_vocab_ids = get_word_ids(sentence_cut, vocabulary, 'orl')
                    sentences_all.append(sentence_vocab_ids)

                else:
                    sentences_original_all.append(' '.join(sentence))
                    sentence_vocab_ids = get_word_ids(sentence, vocabulary, 'orl')
                    m_all.append(m)
                    labels_all.append(labels)
                    sentences_all.append(sentence_vocab_ids)

                ctx_len.append(len(ctx))
                ctx_all.append(ctx)
                ds_all.append(ds_vocab_ids)
                ds_indices_all.append(ds_indices)
                ds_len.append(len(ds_vocab_ids))

                stats_dict['ds_num_after_filter'] += 1
                if not holders and targets:
                    stats_dict['ds_no_holder'] += 1
                if not targets and holders:
                    stats_dict['ds_no_target'] += 1
                if not holders and not targets:
                    stats_dict['ds_no_roles'] += 1
                stats_dict['num_holder_after_filter'] += len(holders)
                stats_dict['num_target_after_filter'] += len(target)

    data_path = 'mpqa_files/' + exp_setup_id + '/'
    data_path = os.path.join(os.path.dirname(__file__), data_path)

    stats_file = open(data_path + 'stats_' + mode + '.txt', 'a')

    att_str = ['sentiment-neg', 'sentiment-pos', 'arguing-pos', 'other-attitude', 'intention-pos', 'arguing-neg',
               'agree-pos', 'speculation', 'agree-neg', 'intention-neg']
    att_count = [type_counts[att] for att in att_str]

    stats_dict = OrderedDict((k, v) for k, v in sorted(stats_dict.items(), key=lambda x: x[0]))

    head_file = open(data_path + 'header.txt', 'w')
    head_file.write('\t'.join(list(stats_dict.keys())) + '\t')
    head_file.write('\t'.join(att_str) + '\n')

    stats_file.write('\t'.join([str(x) for x in stats_dict.values()]) + '\t')
    stats_file.write('\t'.join([str(c) for c in att_count]) + '\n')
    stats_file.close()
    # data_file.close()

    assert len(sentences_all) == len(labels_all) == len(ds_all) == len(ds_len) == len(ctx_all) == len(ctx_len) == len(
        m_all)
    return list(zip(list(sentences_all), list(labels_all), list(ds_all), ds_len, list(ctx_all), ctx_len,
               list(m_all))), ds_indices_all, sentences_original_all



def eval_data_iter(data, batch_size, vocabulary, srl_label_dict):
    '''构建每一次迭代的batch'''
    if len(data) % float(batch_size) == 0:
        num_batches = int(len(data) / batch_size)
    else:
        num_batches = int(len(data) / batch_size) + 1

    batches = []
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data))
        batch = data[start_index:end_index]
        '''取出一个batch_size大小的数据'''

        batch_padded = pad_orl_data(batch, vocabulary)
        '''做好padding，使结构同一'''
        batches.append(batch_padded)

    return batches

def build_train_iter(orl_train, dp_train, batch_size, word2idx, n_epochs):


    orl_train = random.sample(orl_train, len(orl_train))
    #打乱顺序

    dp_train = random.sample(dp_train, len(dp_train))
    '''print(orl_train[0],dp_train[0])'''
    batches = [[],[]]

    for task_id, data in enumerate([dp_train, orl_train]):
        print("building batch")

        if len(data) % float(batch_size) == 0:
            num_batches = int(len(data) / batch_size)
        else:
            num_batches = int(len(data) / batch_size) + 1

        #print("the task and its num_batches is", task_id, num_batches)

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, len(data))

            batch = data[start_index:end_index]
            
            if task_id == 1:
                batch = pad_orl_data(batch, word2idx)

            '''print("the task and the num_batch is", task_id, num_batches, len(data),start_index, end_index)
            '''
            batches[task_id].append(batch)


    batch_return = []
    for it in range(2 * n_epochs):
        task_id = 0 if it%2 == 0 else 1
        #task_id = 1
        randint = random.sample(range(len(batches[task_id])),1)[0]
        batch = batches[task_id][randint]
        print("task_id and batch", task_id, batch[0])
        batch_return.append(batch)

    return batch_return
