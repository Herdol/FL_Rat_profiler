from tokenize import Double
"""from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# if gpu is to be used
device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} Device selected".format(device))
random_seeds = np.random.randint(500, size = 5)
n_cpu = 1


#class CustomPolicy(FeedForwardPolicy):
#    def __init__(self, *args, **kwargs):
#        super(CustomPolicy, self).__init__(*args, **kwargs,
#                                           net_arch=[dict(pi=[128, 128, 128, 128],
#                                                          vf=[128, 128, 128, 128])],
#                                           feature_extraction="mlp")


# Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#env = gym.make('gym_dataOffload:dataCache-v0')
class ReplayMemory(object):
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class DQN(nn.Module):

    def __init__(self, learning_rate=0.01, state_size=47, 
                 action_size=11, hidden_size=128, hidden_size_1=64, hidden_size_2=32, batch_size=20, gamma=0.90):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size_1)
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.output = nn.Linear(hidden_size_2, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = torch.from_numpy(x,dtype=Double)
        
        x = torch.tensor(x).float()
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        output=self.output(x)
        return output

# Utilities
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
#init_screen = get_screen()
#_, _, screen_height, screen_width = init_screen.shape
#
## Get number of actions from gym action space



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.argmax(policy_net(state))
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    if len(memory.buffer) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in torch.tensor(batch.next_state)
                                                if s is not None])
    state_batch = torch.tensor(batch.state).to(device)
    action_batch = torch.tensor(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.view(128,1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.view(128,47)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



# multiprocess environment
if __name__ == "__main__":
    np.random.seed(0) 
    random_seeds = np.random.randint(500, size = 5)
    for ii in range(1):
    # for ii in range(0, len(random_seeds)):
        seed = random_seeds[ii]
        n_cpu = 8
        #env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)])
        env=gym.make('gym_dataCachingCoding:dataCachingCoding-v0')
        n_actions = env.action_space.n

        policy_net = DQN(n_actions).to(device)
        target_net = DQN(n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters(),lr=0.01)
        memory = ReplayMemory(50000)


        steps_done = 0

        # env = gym.make('gym_dataOffload:dataCache-v0')


        #tensorboard_log = "./logs/ppo2_tensorboard_1_18_2"
        #model_name = "ppo2_environment_1_18_1"
        #model = PPO2(CustomPolicy, env, verbose=0,tensorboard_log=tensorboard_log, n_steps = 5000 )
        #model = PPO2.load(model_name, env=env, verbose=0, tensorboard_log=tensorboard_log, n_steps = 5000)
        #print("Model loaded - ", model_name)
        
        # reward_list = []
        # obs = env.reset()
        # i = 0
        # while i < 500:
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones, info = env.step(action)
        #     reward_list.append(rewards[0])
        #     i += 1
        # print(sum(reward_list))
        save_model_name = "DQN_environment_1_18_2"
        #model.learn(total_timesteps=50000)#, seed=seed
        #model.save(save_model_name)
        #print('Model saved.')

        #del model # remove to demonstrate saving and loading
        #
        #model = PPO2.load(save_model_name, env=env, n_steps=5000)

        # Training phase
        num_eps = 150
        for b in range(num_eps):
            print('Training episode:', b)
            reward_list = []
            obs = env.reset()
            
            time_ep = 0.0
            while time_ep < 125:
                start_time = env.time
                #print("time is {}".format(start_time))
                #start_time = env.get_attr('time')[0]
                state=obs
                action = select_action(state)
                
                next_state, reward, dones, info= env.step(action.item())
                
                reward = torch.tensor([reward], device=device)

                #action, _states = model.predict(obs)
                #obs, rewards, dones, info = env.step(action)
                reward_list.append(reward[0])
                #end_time = env.get_attr('time')[0]
                end_time = env.time
                time_elapsed = end_time - start_time
                memory.add([state, action, next_state, reward])
                state=next_state
                optimize_model()
                time_ep += max(time_elapsed, 0)
                # Store the transition in memory
            target_net.load_state_dict(policy_net.state_dict())

        # Validation phase        
        rewards_recv = []
        finished_jobs = 0.0
        finished_job_size = 0.0
        lost_jobs = 0.0
        exceed_deadlines = 0.0
        lost_job_size = 0.0
        exceed_deadline_job_size = 0.0
        num_eps = 24
        finished_jobs_log=[]
        finished_job_size_log=[]
        exceed_deadlines_log=[]
        exceed_deadline_job_size_log=[]
        for b in range(num_eps):
            print('Validation episode:', b)
            reward_list = []
            obs = env.reset()
            
            time_ep = 0.0
            while time_ep < 125:
                start_time = env.time
                #print("time is {}".format(start_time))
                #start_time = env.get_attr('time')[0]
                state=obs
                action = select_action(state)
                
                next_state, reward, dones, info= env.step(action.item())
                
                reward = torch.tensor([reward], device=device)

                #action, _states = model.predict(obs)
                #obs, rewards, dones, info = env.step(action)
                reward_list.append(reward[0])
                #end_time = env.get_attr('time')[0]
                end_time = env.time
                time_elapsed = end_time - start_time
                memory.add([state, action, next_state, reward])
                state=next_state
                #optimize_model()
                time_ep += max(time_elapsed, 0)
                # Store the transition in memory



            rewards_recv.append(np.average(reward_list))
            
            job_history = env.history
            #for ii in range(n_cpu):
            finished_jobs += job_history['completed_jobs']
            finished_job_size += np.sum(job_history['completed_jobs_size'])
            exceed_deadlines += job_history['exceed_deadline']
            exceed_deadline_job_size += np.sum(job_history['exceed_deadline_size'])
            finished_jobs_log.append(finished_jobs)
            finished_job_size_log.append(finished_job_size)
            exceed_deadlines_log.append(exceed_deadlines)
            exceed_deadline_job_size_log.append(exceed_deadline_job_size)
        print('-------------- Print Results---------------------------------------')
        print('Finished jobs: ', finished_jobs/num_eps/n_cpu)
        print('Lost jobs (deadline): ', exceed_deadlines/num_eps/n_cpu)
        print('Average caching rate: ', finished_jobs/(finished_jobs+exceed_deadlines))
        print('Finished job size: ', finished_job_size/num_eps/n_cpu*1)
        print('Lost job size(deadline)', exceed_deadline_job_size/num_eps/n_cpu*1)
        print('Average caching rate(TP): ', finished_job_size/(finished_job_size+exceed_deadline_job_size))

        
        #env.render()
        env.close()
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(finished_jobs_log)
        axs[0, 0].set_title('finished_jobs')
        axs[0, 1].plot(finished_job_size_log, 'tab:orange')
        axs[0, 1].set_title('finished_job_size_log')
        axs[1, 0].plot(exceed_deadlines_log, 'tab:green')
        axs[1, 0].set_title('exceed_deadlines_log')
        axs[1, 1].plot(exceed_deadline_job_size_log, 'tab:red')
        axs[1, 1].set_title('exceed_deadline_job_size_log')
        """plt.plot(finished_jobs_log)
        plt.plot(finished_job_size_log)
        plt.plot(exceed_deadlines_log)
        plt.plot(exceed_deadline_job_size_log)"""
        print('Complete')