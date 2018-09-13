from model.single_model import dp4orl
from build_data import *
from data_util import *
from train_util import train_task_step
from eval_util import eval_orl


def build_orl_train_iter(orl_train, batch_size, word2idx, n_epochs):


    data = random.sample(orl_train, len(orl_train))
    #打乱顺序

    '''print(orl_train[0],dp_train[0])'''
    batches = []


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
           

        batch = pad_orl_data(batch, word2idx)
        batches.append(batch)


    batch_return = []
    for it in range(n_epochs):
        randint = random.sample(range(len(batches)),1)[0]
        batch = batches[randint]
        print("task_id and batch", batch[0])
        batch_return.append(batch)

    return batch_return


def _train_ops(learning_rate, grad_clip):
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
	loss = tf.get_default_graph().get_tensor_by_name('task1/total_loss:0')
	if grad_clip>0:
		grads, vs     = zip(*optimizer.compute_gradients(loss))
		grads, gnorm  = tf.clip_by_global_norm(grads, grad_clip)
		train_op = optimizer.apply_gradients(zip(grads, vs))
	else:
		train_op = optimizer.minimize(loss)
	return train_op


if __name__ == '__main__':

    fold = 0
    word2idx = load_list('word2idx.txt')

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

    train_iter = build_orl_train_iter(orl_train, config.batch_size, word2idx, config.n_epochs)

    orl_train_iter_eval = eval_data_iter(orl_train, config.batch_size, word2idx, None)
    orl_test_iter = eval_data_iter(orl_test, config.batch_size, word2idx, None)
    orl_dev_iter = eval_data_iter(orl_dev, config.batch_size, word2idx, None)

    sess = tf.Session()

    embeddings = np.loadtxt('embedding_matrix.txt')

    model = dp4orl(config, embeddings)


    learning_rate = config.lr
    print("\n\n\n\n\n\n\n\n")
    print("the learning rate is", learning_rate)    
    print("\n\n\n\n\n\n\n\n") 

    init_vars = tf.global_variables_initializer()

    sess.run(init_vars)

    saver = tf.train.Saver()

    holder_flist = [[], [], []]
    target_flist = [[], [], []]
    learning_rate = config.lr

    for it, batch in enumerate(train_iter):

        #print('the type of train ops is', type(train_ops))
        #print('the length of train_op is', len(train_ops))

        train_task_step(model, sess, 1, batch, config, learning_rate)

        if (it+1) % 100 ==0:

            if learning_rate > 0.001:
                learning_rate = learning_rate/2

            print("\n\n\n\n\n\n\n\n")
            print("the learning rate is", learning_rate)
            print("\n\n\n\n\n\n\n\n")        	


            sess.run(init_vars)            
            binary_fscore_train, proportional_fscore_train = eval_orl(orl_train_iter_eval, sess, model, 1)
            binary_fscore_dev, proportional_fscore_dev = eval_orl(orl_dev_iter, sess, model, 1)
            binary_fscore_test, proportional_fscore_test = eval_orl(orl_test_iter, sess, model, 1)


            holder_flist[0].append(proportional_fscore_train[1])
            holder_flist[1].append(proportional_fscore_dev[1])
            holder_flist[2].append(proportional_fscore_test[1])

            target_flist[0].append(proportional_fscore_train[2])
            target_flist[1].append(proportional_fscore_dev[2])
            target_flist[2].append(proportional_fscore_test[2])

            print("binary_fscore_train is", binary_fscore_train, binary_fscore_dev, binary_fscore_test)







