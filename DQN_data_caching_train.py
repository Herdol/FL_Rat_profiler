#%tensorflow_version 1.x
import time
import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import pickle
import trfl
from DQN_utilities import *
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# env parameters
STATE_SIZE_0 = 52
ACTION_SIZE_0 = 5
STATE_SIZE_1 = 36
ACTION_SIZE_1 = 11
RESTORE_PAR = False
ckpt_dir = "./96.9_high_TP_ddl_2_20/checkpoints_2/dataCache.ckpt-999"
tensorboard_log_dir = "./log"
train_summary_writer = tf.summary.FileWriter(tensorboard_log_dir)

# define network hyperparameters
train_episodes = 5000         # max number of episodes to learn from
max_steps = 125                # max steps in an episode
gamma = 0.99                   # future reward discount

# epsilon greedy parameters
epsilon_start = 1.0          # exploration probability at start
epsilon_min = 0.0001            # minimum exploration probability
epsilon_step = (epsilon_start-epsilon_min)/(train_episodes-100)

learning_rate = 1e-4         # Q-network learning rate

# memory parameters
memory_size = int(1e4)            # memory capacity
batch_size = 50                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

# how often in steps to update target network 
update_target_every = 2000

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

# define memory
memory_0 = Memory(max_size=memory_size)
memory_1 = Memory(max_size=memory_size)

saver = tf.train.Saver(max_to_keep=1000)

# target network update op; TRFL way
target_network_update_ops_0 = trfl.update_target_variables(targetQN_0.get_qnetwork_variables(), 
                                                         trainQN_0.get_qnetwork_variables(),tau=1.0)
target_network_update_ops_1 = trfl.update_target_variables(targetQN_1.get_qnetwork_variables(), 
                                                         trainQN_1.get_qnetwork_variables(),tau=1.0)

rewards_list = []
for ii in range(10):
    rewards_list.append((0,0))
loss_list_0 = []
loss_list_1 = []
rews_recent = 0.0

config = tf.ConfigProto(device_count={"CPU":12},
                inter_op_parallelism_threads = 2,
                intra_op_parallelism_threads = 8,
                log_device_placement=True)

with tf.Session(config=config) as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # restore trained parameters
    if RESTORE_PAR:
        saver.restore(sess, ckpt_dir)
        print('parameter restored')
    epsilon = epsilon_start
    step = 0
    step_0 = 0
    step_1 = 0
    summary = tf.Summary()
    for ep in range(1, train_episodes):
        total_reward = 0
        env.reset(USER_USAGE = False)
        state = env.state()
        # train DQN_0
        while env.bs['init'].time < max_steps:                       
            # update target q network
            if step_0 % update_target_every == 0:
                #TRFL way
                sess.run(target_network_update_ops_0)
                # print("\nTarget network updated.")

            # epsilon greedy exploration
            if np.random.rand() <= epsilon:
                # Make a random action
                action_0 = env.action_space_0.sample()
            else:
                # Get action from Q-network
                feed = {trainQN_0.input_: np.reshape(state,(1,len(state)))}
                action_values = sess.run(trainQN_0.output, feed_dict=feed)
                action_0 = np.argmax(action_values)            
            # get another action through Q network
            state_1 = env.state_1(action_0)
            feed = {trainQN_1.input_: np.reshape(state_1,(1,len(state_1)))}
            action_values = sess.run(trainQN_1.output, feed_dict=feed)
            action_1 = np.argmax(action_values)  
            # take action
            next_state, reward, info, done = env.step([action_0, action_1])
            total_reward += reward
            # add the last experience to memory
            memory_0.add((state, action_0, reward, next_state))
            state = next_state
            step_0 += 1

            if len(memory_0.buffer) > batch_size:
                # sample mini-batch from memory
                batch = memory_0.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                # train network
                target_Qs = sess.run(targetQN_0.output, feed_dict={targetQN_0.input_: next_states})
                # calculate td_error within TRFL
                loss, _ = sess.run([trainQN_0.loss, trainQN_0.opt],
                                    feed_dict={trainQN_0.input_: states,
                                            trainQN_0.targetQ_: target_Qs,
                                            trainQN_0.reward_: rewards,
                                            trainQN_0.action_: actions})
                loss_list_0.append(loss)
                step += 1
        
        # train DQN_1
        while env.bs['init'].time < max_steps*2:                       
            # update target q network
            if step_1 % update_target_every == 0:
                #TRFL way
                sess.run(target_network_update_ops_1)
                # print("\nTarget network updated.")

            # get action_0 from Q network
            feed = {trainQN_0.input_: np.reshape(state,(1,len(state)))}
            action_values = sess.run(trainQN_0.output, feed_dict=feed)
            action_0 = np.argmax(action_values)            
            state_1 = env.state_1(action_0)

            # epsilon greedy exploration
            if np.random.rand() <= epsilon:
                # Make a random action
                action_1 = env.action_space_1.sample()
            else:
                # Get action from Q-network
                feed = {trainQN_1.input_: np.reshape(state_1,(1,len(state_1)))}
                action_values = sess.run(trainQN_1.output, feed_dict=feed)
                action_1 = np.argmax(action_values)            

            # take action
            next_state, reward, info, done = env.step([action_0, action_1])
            next_state = env.state_1(action_0)
            total_reward += reward
            # add the last experience to memory
            memory_1.add((state_1, action_1, reward, next_state))
            state_1 = next_state
            step_1 += 1

            if len(memory_1.buffer) > batch_size:
                # sample mini-batch from memory
                batch = memory_1.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                # train network
                target_Qs = sess.run(targetQN_1.output, feed_dict={targetQN_1.input_: next_states})
                # calculate td_error within TRFL
                loss, _ = sess.run([trainQN_1.loss, trainQN_1.opt],
                                    feed_dict={trainQN_1.input_: states,
                                            trainQN_1.targetQ_: target_Qs,
                                            trainQN_1.reward_: rewards,
                                            trainQN_1.action_: actions})
                loss_list_1.append(loss)
                step += 1
            
        saver.save(sess, "checkpoints_2/dataCache.ckpt", global_step = ep)    
        rewards_list.append((ep, total_reward))

        # Add summaries to tensorboard
        summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=total_reward)])
        train_summary_writer.add_summary(summary, global_step=ep)

        # the current episode ends
        if ep % 20 == 0:
            print('Episode: {}'.format(ep),
                'Total reward in last 20 eps: {}'.format(np.mean(np.array(rewards_list[-20:])[:,1])),
                'Epsilon: {:.4f}'.format(epsilon),
                'Global step: {}'.format(step))

            with open('rewards_list.txt', 'wb') as fp:
                pickle.dump(rewards_list, fp)
            with open('losses_list_0.txt', 'wb') as fp:
                pickle.dump(loss_list_0, fp)
            with open('losses_list_1.txt', 'wb') as fp:
                pickle.dump(loss_list_1, fp)

        # # start new episode
        # env.reset(USER_USAGE = False)
        # state = env.state()
        # reduce epsilon
        epsilon -= epsilon_step
        if epsilon < epsilon_min:
            epsilon = epsilon_min

    saver.save(sess, "checkpoints_2/dataCache_end.ckpt")