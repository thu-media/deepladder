import multiprocessing as mp
import numpy as np
import logging
import os
import sys
# import network_env as NetworkEnv
import network_env as NetworkEnv
import meppo as network
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [4, 20]
#[4, 224, 224, 3]
A_DIM = 21
N_DIM = 6
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 500  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 500
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './deepladder-10'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
#'./cooked_traces/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './deepladder-10/log'
AC_BATCH = 5
PPO_TRAINING_EPO = 10
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None #'./deepladder/nn_model_ep_36000.ckpt'    

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python rl_test.py ' + nn_model)

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(reward[-1])
        entropies.append(np.mean(entropy))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    config=tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5)
    config.gpu_options.allow_growth = True
    with tf.device('/gpu:0'):
        with tf.Session(config = config) as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
            summary_ops, summary_vars = build_summaries()

            actor = network.Network(sess,
                    state_dim=S_DIM, action_dim=A_DIM,
                    n_dim=N_DIM,
                    learning_rate=ACTOR_LR_RATE)

            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
            saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

            # restore neural net parameters
            nn_model = NN_MODEL
            if nn_model is not None:  # nn_model is the path to file
                saver.restore(sess, nn_model)
                print("Model restored.")

            max_reward, max_epoch = -10000., 0
            tick_gap = 0
            
            # while True:  # assemble experiences from agents, compute the gradients
            for epoch in range(TRAIN_EPOCH):
                # synchronize the network parameters of work agent
                actor_net_params = actor.get_network_params()
                for i in range(NUM_AGENTS):
                    net_params_queues[i].put(actor_net_params)

                m, n, a, p, g = [], [], [], [], []
                for i in range(NUM_AGENTS):
                    m_, n_, a_, p_, g_ = exp_queues[i].get()
                    m += m_
                    n += n_
                    a += a_
                    p += p_
                    g += g_
                m_batch = np.stack(m, axis=0)
                n_batch = np.stack(n, axis=0)
                a_batch = np.vstack(a)
                p_batch = np.vstack(p)
                v_batch = np.vstack(g)
                
                for _ in range(PPO_TRAINING_EPO):
                    actor.train(m_batch, n_batch, a_batch, p_batch, v_batch, epoch)
                # actor.train(s_batch, a_batch, v_batch, epoch)
                
                if epoch % MODEL_SAVE_INTERVAL == 0:
                    # Save the neural net parameters to disk.
                    save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                        str(epoch) + ".ckpt")
                    avg_reward, avg_entropy = testing(epoch,
                        SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                        test_log_file)
                    
                    if avg_reward > max_reward:
                        max_reward = avg_reward
                        max_epoch = epoch
                        tick_gap = 0
                    else:
                        tick_gap += 1
                    
                    if tick_gap >= 5:
                        # saver.restore(sess, SUMMARY_DIR + "/nn_model_ep_" + str(max_epoch) + ".ckpt")
                        actor.set_entropy_decay()
                        tick_gap = 0

                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: actor.get_entropy(epoch),
                        summary_vars[1]: avg_reward,
                        summary_vars[2]: avg_entropy
                    })
                    writer.add_summary(summary_str, epoch)
                    writer.flush()

def agent(agent_id, net_params_queue, exp_queue):
    env = NetworkEnv.NetworkEnvDiscrete(random_seed=agent_id, a_dim=N_DIM)
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
            actor = network.Network(sess,
                                    state_dim=S_DIM, action_dim=A_DIM,
                                    n_dim=N_DIM,
                                    learning_rate=ACTOR_LR_RATE)

            # initial synchronization of the network parameters from the coordinator
            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

            for epoch in range(TRAIN_EPOCH):
                m, n, a, p, v = [], [], [], [], []
                for _ in range(AC_BATCH):
                    m_batch, n_batch, a_batch, p_batch, r_batch = [], [], [], [], []
                    r_batch = [0.]

                    obs = env.reset()
                    for step in range(TRAIN_SEQ_LEN):
                        m_batch.append(obs['network'])
                        n_batch.append(obs['video'])
                        action_prob = actor.predict(obs)
                        
                        # gumbel noise
                        mask = obs['network'][2, :]
                        # Hint: how many '1' in the mask?
                        action_prob_mask = action_prob[np.where(mask > 0)]
                        noise_mask = np.random.gumbel(size=len(action_prob_mask))
                        act_mask = np.argmax(np.log(action_prob_mask) + noise_mask)
                        # Taken to your actions now.
                        act = np.where(mask > 0)[0][act_mask]
                        
                        assert mask[act] > 0.

                        obs, rew, done, info = env.step(act)

                        log_file.write(str(epoch) + '\t' +
                                    str(action_prob) + '\t' +
                                    str(act_mask) + '\t' +
                                    str(action_prob_mask) + '\t' +
                                    str(act) + '\t' +
                                    str(rew) + '\t' + '\n')
                        log_file.flush()

                        assert action_prob[act] > 0.
                        action_vec = np.zeros(A_DIM)
                        action_vec[act] = 1
                        
                        a_batch.append(action_vec)
                        r_batch.append(rew)
                        p_batch.append(action_prob)

                        if done:
                            break
                    r_batch = np.diff(r_batch)
                    v_batch = actor.compute_v(m_batch, n_batch, a_batch, r_batch, done)
                    m += m_batch
                    n += n_batch
                    a += a_batch
                    p += p_batch
                    v += v_batch
                exp_queue.put([m, n, a, p, v])

                actor_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("Beta", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [td_loss, eps_total_reward, entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
