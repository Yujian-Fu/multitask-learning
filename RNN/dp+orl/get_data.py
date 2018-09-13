from config import config
from build_data import *
from data_util import *
import numpy as np


def get_train_batch(fold):
    word2idx = load_list('word2idx.txt')
    embedding_dict = load_list('embedding_dict.txt')
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

    highlight("transform orl_train_data")
    orl_train, _, _ = transform_orl_data(orl_train_corpus, word2idx, config.window_size, 'train',
                                            config.exp_setup_id, config.att_link_obligatory)

    highlight("transform orl_test_data")
    orl_test, _, _ = transform_orl_data(orl_test_corpus, word2idx, config.window_size, 'test',
                                            config.exp_setup_id, config.att_link_obligatory)

    highlight("transform orl_dev_data")
    orl_dev, _, _ = transform_orl_data(orl_dev_corpus, word2idx, config.window_size, 'dev',
                                           config.exp_setup_id, config.att_link_obligatory)

    highlight("transform dp_train_data")
    dp_train = transform_dp_data(dp_train_corpus, embedding_dict, word2idx)
    #dp_train, dp_target = transform_dp_data(dp_train_corpus,word2idx,pos2idx,dep2idx)
        
    dp_train_ = []
    for sent in dp_train:
        for word_com in sent:
            dp_train_.append(word_com)
        

    highlight("built iter")
    train_iter = build_train_iter(orl_train, dp_train_, config.batch_size, word2idx, config.n_epochs)

    orl_train_iter_eval = eval_data_iter(orl_train, config.batch_size, word2idx, None)
    orl_test_iter = eval_data_iter(orl_test, config.batch_size, word2idx, None)
    orl_dev_iter = eval_data_iter(orl_dev, config.batch_size, word2idx, None)

    return train_iter, orl_train_iter_eval, orl_test_iter, orl_dev_iter



