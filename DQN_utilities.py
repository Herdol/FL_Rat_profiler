import gym
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
from collections import deque
import trfl

class QNetwork:
    def __init__(self, name, learning_rate=0.01, state_size=20, 
                 action_size=10, hidden_size=128, hidden_size_1=64, hidden_size_2=32, batch_size=20, gamma=0.90):
        with tf.variable_scope(name):
            #set up tensors
            self.input_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            self.action_ = tf.placeholder(tf.int32, [batch_size], name='actions')
            self.targetQ_ = tf.placeholder(tf.float32, [batch_size, action_size], name='target')
            self.reward_ = tf.placeholder(tf.float32,[batch_size],name="reward")
            self.discount_ = tf.constant(gamma,shape=[batch_size],dtype=tf.float32,name="discount")
            self.name = name
   
            #This is the Neural Network. For CNN implementation, the input would feed into a CNN layer prior to these layers.
            self.fc1 = tf.contrib.layers.fully_connected(self.input_, hidden_size, activation_fn=tf.nn.relu)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size_1, activation_fn=tf.nn.relu)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size_2, activation_fn=tf.nn.relu)
            self.fc4 = tf.contrib.layers.fully_connected(self.fc3, hidden_size_2, activation_fn=tf.nn.relu)
            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc4, action_size, activation_fn=None)

            #TRFL qlearning
            qloss, q_learning = trfl.qlearning(self.output, self.action_, self.reward_, self.discount_, self.targetQ_)
            self.loss = tf.reduce_mean(qloss)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]