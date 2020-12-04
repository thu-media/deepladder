import warnings
warnings.filterwarnings("ignore")
import numpy as np
import logging
import os
import sys
import fixed_network_env as f_env
import meppo as network
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [4, 20]
#[4, 224, 224, 3]
A_DIM = 20
N_DIM = 6
ACTOR_LR_RATE = 1e-4
RANDOM_SEED = 42
RAND_RANGE = 1000
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]
TRAIN_SEQ_LEN = 300
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './oboe_traces/'
# TEST_TRACES = './hd-fs-test/'
#'./cooked_test_traces/'

def main():

    np.random.seed(RANDOM_SEED)
    
    env = f_env.FixedNetworkEnvDiscrete(cooked_trace_folder=TEST_TRACES)

    with tf.Session() as sess:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                n_dim=N_DIM,
                                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")
        # fw = open('test_results.log', 'w')
        rews = []
        vmafs, utils, sizes = [], [], []
        while env.next():
            log_file = open(LOG_FILE + '_' + env.get_trace_name(), 'w')
            obs = env.reset()
            for step in range(TRAIN_SEQ_LEN):
                action_prob = actor.predict(obs)
                entropy_ = 0. - np.dot(action_prob, np.log(action_prob))
                
                # gumbel noise
                
                noise = np.random.gumbel(size=len(action_prob))
                act = np.argmax(np.log(action_prob) + noise)
                        
                obs, rew, done, info = env.step(act)
                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(env.videos.get_video_name()) + '\t' +
                            str(info['vmaf']) + '\t' +
                            str(info['util']) + '\t' +
                            str(info['size']) + '\t' +
                            str(act) + '\t' +
                            str(entropy_) + '\t' + 
                            str(rew) + '\n')
                log_file.flush()
                if done:
                    rews.append(rew)
                    vmafs.append(info['vmaf'])
                    utils.append(info['util'])
                    sizes.append(info['size'])
                    break
            log_file.close()
        print(np.mean(vmafs), np.mean(utils), np.mean(sizes))
        print(np.std(vmafs), np.std(utils), np.std(sizes))

if __name__ == '__main__':
    main()
