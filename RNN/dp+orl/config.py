import os

class config():
    embedding_trainable = 'False'

    window_size = 2
    embedding_name = 'embedding_matrix.npz'
    word2idx_name = 'word2idx.txt'
    embedding_dict_name = 'embedding_dict.txt'
    n_classes_dp = 2
    n_classes_orl = 8
    seed = 24

    hidden_size = 100
    n_layers_shared = 3
    keep_rate_input = 0.7
    keep_rate_output = 0.85
    keep_state_rate = 0.85
    out_dir = 'outputs'

    opt = 'adam'
    lr = 0.1
    adv = 0
    batch_size = 32
    n_epochs = 20000
    grad_clip = 0

    model = 'fs'
    emb_dim = 300

    exp_setup_id = 'new'
    att_link_obligatory = 'false'
    reg_coef = 0.5
    lr_decay = 0.5