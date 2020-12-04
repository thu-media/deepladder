import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-6
GAMMA = 0.99
# PPO2
EPS = 0.2

class Network():

    def masked_softmax_v2(self, logits, mask):
        """
        Masked softmax over dim 1
        :param logits: (N, L)
        :param mask: (N, L)
        :return: probabilities (N, L)
        """
        indices = tf.where(mask)
        values = tf.gather_nd(logits, indices)
        denseShape = tf.cast(tf.shape(logits), tf.int64)
        sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
        result = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
        result.set_shape(logits.shape)
        return result

    def CreateNetwork(self, network_inputs, video_inputs):
        with tf.variable_scope('actor'):
            # network features [None, 3, 15]
            # [None, 3, 128]
            n_net = tflearn.conv_1d(network_inputs[:, :3, :], FEATURE_NUM, 3, activation='relu')

            # video_features [None, 4, 2048]
            v_net = tflearn.conv_1d(video_inputs, FEATURE_NUM, 4, activation='relu')
            
            a_net = tflearn.fully_connected(network_inputs[:, 3:4, -1:], FEATURE_NUM, activation='relu')

            split_n = tflearn.flatten(n_net)
            split_v = tflearn.flatten(v_net)
            split_a = tflearn.flatten(a_net)

            merge_net = tflearn.merge([split_n, split_v, split_a], 'concat')
            #tf.concat([v_net, n_net], axis=1)
            # merge_net = tflearn.conv_1d(merge_net, 128, 3, activation='relu')

            pi_net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')
            value_net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')
            
            pi_value = tflearn.fully_connected(pi_net, self.a_dim, activation='linear')
            mask = network_inputs[:, 2, :]
            pi = self.masked_softmax_v2(pi_value, mask)
            pi = tf.clip_by_value(pi, ACTION_EPS, 1 - ACTION_EPS)

            value = tflearn.fully_connected(value_net, 1, activation='linear')
            return pi, value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)

    def __init__(self, sess, state_dim, action_dim, n_dim, learning_rate):
        self._entropy = 10. #0.3277
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.n_dim = n_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.network_inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.video_inputs = tf.placeholder(tf.float32, [None, 4, 2048])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)
        self.pi, self.val = self.CreateNetwork(network_inputs=self.network_inputs, \
            video_inputs=self.video_inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.entropy = tf.multiply(self.real_out, tf.log(self.real_out))
        self.adv = tf.stop_gradient(self.R - self.val)
        self.ratio = self.r(self.real_out, self.old_pi, self.acts)
        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_pi, self.acts) * self.adv, 
                            tf.clip_by_value(self.r(self.real_out, self.old_pi, self.acts), 1 - EPS, 1 + EPS) * self.adv
                        )
        # https://arxiv.org/pdf/1912.09729.pdf
        self.dualppo = tf.cast(tf.less(self.adv, 0.), dtype=tf.float32)  * tf.maximum(self.ppo2loss, 3. * self.adv) + \
            tf.cast(tf.greater_equal(self.adv, 0.), dtype=tf.float32) * self.ppo2loss
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss = - tf.reduce_sum(self.dualppo) \
            + self.entropy_weight * tf.reduce_sum(self.entropy)
            
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
        self.val_loss = tflearn.mean_square(self.val, self.R)
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate * 10.).minimize(self.val_loss)
        
    def predict(self, obs):# obs_network, obs_video):
        obs_network, obs_video = obs['network'], obs['video']
        obs_n = np.reshape(obs_network, (1, self.s_dim[0], self.s_dim[1]))
        obs_v = np.reshape(obs_video, (1, 4, 2048))
        action = self.sess.run(self.real_out, feed_dict={
            self.network_inputs: obs_n,
            self.video_inputs: obs_v
        })
        return action[0]

    def set_entropy_decay(self, decay = 0.8):
        self._entropy *= decay
        self._entropy = np.clip(self._entropy, 1e-10, 10.0)

    def set_entropy(self, entropy):
        self._entropy = entropy

    def get_entropy(self, step):
        return self._entropy

    def train(self, m_batch, n_batch, a_batch, p_batch, v_batch, epoch):
        m_batch, n_batch, a_batch, p_batch, v_batch = tflearn.data_utils.shuffle(m_batch, n_batch, \
            a_batch, p_batch, v_batch)
        self.sess.run([self.optimize, self.val_opt, self.ratio], feed_dict={
            self.network_inputs: m_batch,
            self.video_inputs: n_batch,
            self.acts: a_batch,
            self.R: v_batch, 
            self.old_pi: p_batch,
            self.entropy_weight: self.get_entropy(epoch)
        })

    def compute_v(self, m_batch, n_batch, a_batch, r_batch, terminal):
        ba_size = len(m_batch)
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = r_batch[-1]  # terminal state
        else:    
            v_batch = self.sess.run(self.val, feed_dict={
                self.network_inputs: m_batch,
                self.video_inputs: n_batch
            })
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)
