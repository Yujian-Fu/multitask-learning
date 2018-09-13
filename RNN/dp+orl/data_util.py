import tensorflow as tf
import argparse
import logging
import json
import random
import os
import numpy as np
from collections import Counter
import itertools
from config import config

NULL = "<null>"
UNK = "<unk>"
ROOT = "<root>"
PAD = "<pad>"
EMP = "<empty>"


def highlight(info):
    print(80*"=")
    print(info)
    print(80*"=")


def save_list(filename, content, mode = 'w'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    content = json.dumps(content)
    file.write(content)
    file.close()

def load_list(filename, mode = 'r'):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    file = open(filename, mode)
    content = file.read()
    content = json.loads(content)
    file.close()
    return content

def save_array(filename,file):
    np.savez_compressed(filename, file)

def load_array(filename):
    return np.load(filename)

def element2id(vocab):
    item2id = {}
    idx = 0
    for item in vocab:
        item2id[item] = idx
        idx += 1
    item2id[PAD] = idx
    idx += 1
    item2id[UNK] = idx
    idx += 1
    item2id[NULL] = idx
    idx += 1
    item2id[ROOT] = idx
    idx += 1
    item2id[EMP] = idx

    return item2id

def build_vocab(sentence_orl, sentence_dp):
    #with open ('glove.6B.300d.txt') as f:
    #all words in dp
    word_dp = set((w[1] for s in sentence_dp for w in s if w))
    #all words in orl
    word_orl_count = dict(Counter(itertools.chain(*sentence_orl)).most_common())
    word_orl_prune = {k: v for k, v in word_orl_count.items()}
    word_orl_list = zip(word_orl_prune.keys(), word_orl_prune.values())
    word_orl = set([x[0] for x in word_orl_list])

    all_words = word_dp | word_orl
    print("all words", len(all_words),type(all_words))

    word_vocab = list(all_words)

    word2idx = element2id(word_vocab)#43141
    idx2word = {idx:word for (word,idx) in word2idx.items()}


    return all_words, word2idx, idx2word

def build_embedding(word2idx, all_words):

    vocab_size = len(word2idx)
    embedding_dict = {}
    embedding_matrix = np.zeros((vocab_size,config.emb_dim),dtype = np.float32)

    with open ('glove.6B.300d.txt') as f:
        for line in f:
            line = line.split()
            if line[0] in all_words:
                embedding_dict[line[0]] =  list(([float(n) for n in line[1:]]))

    vec_sum = list([0]*300)
    embedding_dict[PAD] = vec_sum
    embedding_dict[NULL] = vec_sum


    for word in word2idx:
        if word in embedding_dict:
            embedding_matrix[word2idx[word],:] = embedding_dict[word]
            print('the',word2idx[word], 'th line is:', embedding_matrix[word2idx[word],:])


    embedding_matrix[word2idx[UNK],:] = np.mean(embedding_matrix, axis = 0).reshape(1,config.emb_dim)
    print('the UNK of embedding_matrix is', word2idx[UNK], embedding_matrix[word2idx[UNK],:])

    embedding_dict[UNK] = embedding_matrix[word2idx[UNK],:]

    for word in word2idx:
        if word not in embedding_dict:
            embedding_matrix[word2idx[word],:] = embedding_matrix[word2idx[UNK],:]



    embedding_matrix[word2idx[NULL]] = embedding_dict[NULL]
    embedding_matrix[word2idx[PAD]] = embedding_dict[PAD]

    return embedding_dict, embedding_matrix
