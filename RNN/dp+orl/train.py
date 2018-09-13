from config import config
from get_data import *
from data_util import *
from train_util import _train_ops, train_task_step
from eval_util import *

if __name__ == '__main__':

    for fold in range(4):
        train_iter, orl_train_iter_eval, orl_test_iter, orl_dev_iter = get_train_batch(fold)

        if config.model in ['asp','sd']:
            from model.sp_mtl_models_v2 import dp4orl

        if config.model == 'fs':
            logging.info('loading fs model')
            from model.fs_mtl_model import dp4orl

        if config.model == 'hmtl':
            logging.info('loading hmtl model')
            from model.hmtl_model import dp4orl

        sess = tf.Session()

        embeddings = np.loadtxt('embedding_matrix.txt')

        model = dp4orl(config, embeddings)

        learning_rate = config.lr

        train_ops = _train_ops(learning_rate, config.grad_clip)

        init_vars = tf.global_variables_initializer()

        sess.run(init_vars)

        saver = tf.train.Saver()

        holder_flist = [[], [], []]
        target_flist = [[], [], []]
        learnig_rate = config.lr

        for it, batch in enumerate(train_iter):

            task_id = int(it%2)
        #print('the type of train ops is', type(train_ops))
        #print('the length of train_op is', len(train_ops))

            train_task_step(model, sess, task_id, batch, config, learning_rate)

            if (it+1) % 100 ==0:

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

                fig_path = config.out_dir + 'orl/figs/holder' + str(fold+1) + '/'
                if not os.path.exists(fig_path):
            	    os.makedirs(fig_path)
                plot_training_curve(fig_path + 'learning_curve.png', int((it+1)/100),holder_flist)

                fig_path = config.out_dir + 'orl/figs/target/' + str(fold+1) + '/'
                if not os.path.exists(fig_path):
            	    os.makedirs(fig_path)
                plot_training_curve(fig_path + 'learning_curve.png', int((it+1)/100), target_flist)






    		
