import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import trfl
from DQN_utilities import *

# env parameters
STATE_SIZE_0 = 52
ACTION_SIZE_0 = 5
STATE_SIZE_1 = 36
ACTION_SIZE_1 = 11
RESTORE_PAR = False
ckpt_dir = "./checkpoints_2/dataCache.ckpt-4999"

# define network hyperparameters
validation_episodes = 200         # number of episodes
max_steps = 125                # max steps in an episode
gamma = 0.95                   # future reward discount

learning_rate = 1e-4         # Q-network learning rate

# memory parameters
memory_size = int(1e8)            # memory capacity
batch_size = 50                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

#declare TRFL in graph
tf.reset_default_graph()
# the DQN to choose a vehicle
trainQN_0 = QNetwork(state_size=STATE_SIZE_0, action_size=ACTION_SIZE_0, name='train_qn_0',learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
targetQN_0 = QNetwork(state_size=STATE_SIZE_0, action_size=ACTION_SIZE_0, name='target_qn_0',learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
# the DQN to choose a job and RAT
trainQN_1 = QNetwork(state_size=STATE_SIZE_1, action_size=ACTION_SIZE_1, name='train_qn_1',learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
targetQN_1 = QNetwork(state_size=STATE_SIZE_1, action_size=ACTION_SIZE_1, name='target_qn_1',learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)

# Initialize the simulation
env = gym.make('gym_dataOffload:dataCache-v0')
env.reset(USER_USAGE = False)
state = env.state()

rewards_list = []
loss_list = []
saver = tf.train.Saver()
# finished jobs and lost jobs for all eps
finished_jobs = []
lost_jobs = []

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # restore trained parameters
    saver.restore(sess, ckpt_dir)
    step = 0
    for ep in range(1, validation_episodes):
        t = 0
        total_reward = 0
        env.reset(USER_USAGE = False)
        state = env.state()        
        while env.bs['init'].time < max_steps:
            # Get action_0 from Q-network
            feed = {trainQN_0.input_: np.reshape(state,(1,len(state)))}
            action_values = sess.run(trainQN_0.output, feed_dict=feed)
            action_0 = np.argmax(action_values)
            # Get action_1 from Q-network
            state_1 = env.state_1(action_0)
            feed = {trainQN_1.input_: np.reshape(state_1,(1,len(state_1)))}
            action_values = sess.run(trainQN_1.output, feed_dict=feed)
            action_1 = np.argmax(action_values)            
            # take action
            next_state, reward, info, done = env.step([action_0,action_1])
            total_reward += reward
            state = next_state
            t += 1       
        
        if ep % 20 == 0:
            print('Episode: {}'.format(ep),
                'Total reward: {}'.format(total_reward))
        rewards_list.append((ep, total_reward))

        # record finished jobs and lost jobs in this ep
        cur_finished, cur_lost_r, cur_lost_d = env.return_jobs()
        finished_jobs.append(cur_finished)
        lost_jobs.append([cur_lost_r, cur_lost_d])

        # start new episode
        env.reset(USER_USAGE = False)
        state = env.state()

# with tf.Session() as sess:
#     # Initialize variables
#     sess.run(tf.global_variables_initializer())
#     # restore trained parameters
#     saver.restore(sess, ckpt_dir)
#     step = 0
#     for ep in range(1, validation_episodes):
#         t = 0
#         total_reward = 0        
#         while t < max_steps:
#             # Get action_0 from Q-network
#             feed = {trainQN_0.input_: np.reshape(state,(1,len(state)))}
#             action_values = sess.run(trainQN_0.output, feed_dict=feed)
#             action_0 = np.argmax(action_values)
#             # Get action_1 from Q-network
#             state_1 = env.state_1(action_0)
#             feed = {trainQN_1.input_: np.reshape(state_1,(1,len(state_1)))}
#             action_values = sess.run(trainQN_1.output, feed_dict=feed)
#             action_1 = np.argmax(action_values)            
#             # take action
#             next_state, reward, info, done = env.step([action_0,action_1])
#             total_reward += reward
#             state = next_state
#             t += 1       
        
#         if ep % 20 == 0:
#             print('Episode: {}'.format(ep),
#                 'Total reward: {}'.format(total_reward))
#         rewards_list.append((ep, total_reward))

#         # record finished jobs and lost jobs in this ep
#         cur_finished, cur_lost_r, cur_lost_d = env.return_jobs()
#         finished_jobs.append(cur_finished)
#         lost_jobs.append([cur_lost_r, cur_lost_d])

#         # start new episode
#         env.reset(USER_USAGE = False)
#         state = env.state()
    
with open('finished_jobs_DQN.txt', 'wb') as fp:
    pickle.dump(finished_jobs, fp)
with open('lost_jobs_DQN.txt', 'wb') as fp:
    pickle.dump(lost_jobs, fp)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
