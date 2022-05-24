## Update notes ##
# W&B and Sweep module implementation for hyperparameter tuning. 17/03/22
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
import wandb



os.environ['KMP_DUPLICATE_LIB_OK']='True'
## Example W%B commands 

#wandb.config = {
#  "learning_rate": 0.001,
#  "epochs": 100,
#  "batch_size": 128
#}
#wandb.log({"loss": loss})
#
## Optional
#wandb.watch(model)
hyperparameter_defaults = dict(
    BATCH_SIZE = 128,
    GAMMA = 0.96,
    EPS_START = 0.9,
    EPS_END = 0.05,
    EPS_DECAY = 400,
    channels_one = 32,
    channels_two = 64,
    learning_rate = 0.001,
    episodes = 24,
    Replay_memory = 10000)
    
wandb.init(project="RAT_env_initial_tests_v02",config=hyperparameter_defaults, entity="herdol")
config = wandb.config
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
    def __init__(self, max_size = config.Replay_memory):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def wipe_memory(self,max_size=config.Replay_memory):
        self.buffer = deque(maxlen=max_size)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class DQN(nn.Module):

    def __init__(self, learning_rate=config.learning_rate, state_size=47, 
                 action_size=11, hidden_size=config.channels_two, hidden_size_1=config.channels_two, hidden_size_2=config.channels_one, batch_size=config.BATCH_SIZE, gamma=config.GAMMA):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, hidden_size_1)
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        #self.fc4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.output = nn.Linear(hidden_size_2, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = torch.from_numpy(x,dtype=Double)
        
        x = torch.tensor(x).float()
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = self.fc4(x)
        #x = F.relu(x)
        output=self.output(x)
        return output


# Utilities
BATCH_SIZE = config.BATCH_SIZE
GAMMA = config.GAMMA
EPS_START =config.EPS_START 
EPS_END = config.EPS_END
EPS_DECAY =config.EPS_DECAY 


## Get number of actions from gym action space
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if Training==1:
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return torch.argmax(policy_net(state))
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
         with torch.no_grad():
            return torch.argmax(target_net(state)) # was policy_net


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
    non_final_next_states = torch.cat([s for s in torch.tensor(np.array(batch.next_state))
                                                if s is not None])
    state_batch = torch.tensor(batch.state).to(device)
    action_batch = torch.tensor(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.view(BATCH_SIZE,1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.view(BATCH_SIZE,47)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    """for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)"""
    optimizer.step()


def Heuristic_select_action(state):
    view_buffer = state[30:45] # Extract buffer info from state
    
    destinations = view_buffer[1:-1:3] # Extract destinations from buffer
    deadlines= view_buffer[2::3]
    #dest_line=np.argmin(deadlines) # Choose the least remaining time in deadlines
    #dest = destinations[dest_line] * 4 # denormalize veh index
    destinations=destinations * 4
    dest = np.random.choice(destinations,size=1) # Choose random action from destinations
    LTE_available = state[45] 
    mmWave_available = state[46]
    Distances=env.vehicle_positions(scale = False)
    Distances=Distances-500 # distance to the center
    Dist=[]
    for i in range(5):
        x,y=Distances[i][0],Distances[i][1]
        Dist.append(np.sqrt(np.sum((x - y) ** 2, axis=0)))
    # If mmWave available, select it. Because it has more datarate.
    chosen_veh=999
    if mmWave_available == 1:
        for veh in destinations:
            if Dist[int(veh)]<=200:
                action = veh*2 + 1
                chosen_veh=veh
        if chosen_veh>5 and LTE_available == 1: # If all vehicles are out of mmWave coverage # Not clever conditions
            action = dest *2
        elif chosen_veh>5 and LTE_available == 0: 
            action = 10
    elif mmWave_available == 0 and LTE_available == 1 : # the mmWave max range is 200m
        action = dest *2
    else:
        action = 10
        # action = np.random.randint(1,10)
    return int(action)

# Beginning of Sim
if __name__ == "__main__":
    np.random.seed(1) 
    
    random_seeds = np.random.randint(500, size = 5)
    for ii in range(1): # This loop will be used for FL UEs
        # for ii in range(0, len(random_seeds)):
        seed = random_seeds[ii]
        n_cpu = 1
        #env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)])
        env=gym.make('gym_dataCachingCoding:dataCachingCoding-v0')
        n_actions = env.action_space.n

        policy_net = DQN(n_actions).to(device)
        target_net = DQN(n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        model = policy_net
        wandb.watch(model)  

        optimizer = optim.RMSprop(policy_net.parameters(),lr=config.learning_rate)
        memory = ReplayMemory(config.Replay_memory)
        steps_done=0

        # env = gym.make('gym_dataOffload:dataCache-v0')

        #save_model_name = "DQN_environment_1_18_2"
        
        Training=1
        Training_history=env.history
        Training_reward=env.reward()
        
        rewards_recv = []
        Training_finished_jobs = 0.0
        Training_finished_job_size = 0.0
        Training_lost_jobs = 0.0
        Training_exceed_deadlines = 0.0
        Training_lost_job_size = 0.0
        Training_exceed_deadline_job_size = 0.0
        Training_finished_jobs_log=[]
        Training_finished_job_size_log=[]
        Training_exceed_deadlines_log=[]
        Training_exceed_deadline_job_size_log=[]
        # Training phase

        num_eps = 30
        for b in range(config.episodes):
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
                reward_list.append(reward[0])
                
                #end_time = env.get_attr('time')[0]
                end_time = env.time
                time_elapsed = end_time - start_time
                memory.add([state, action, next_state, reward])
                obs=next_state
                optimize_model()
                time_ep += max(time_elapsed, 0)
                # Store the transition in memory
            target_net.load_state_dict(policy_net.state_dict())
            job_history = env.history
            ## Logging ##
            Training_finished_jobs += job_history['completed_jobs']
            Training_finished_job_size += np.sum(job_history['completed_jobs_size'])
            Training_exceed_deadlines += job_history['exceed_deadline']
            Training_exceed_deadline_job_size += np.sum(job_history['exceed_deadline_size'])            

            Training_finished_jobs_log.append(job_history['completed_jobs'])
            Training_finished_job_size_log.append(np.sum(job_history['completed_jobs_size']))
            Training_exceed_deadlines_log.append(job_history['exceed_deadline'])
            Training_exceed_deadline_job_size_log.append(np.sum(job_history['exceed_deadline_size']))
            metrics = {'Training_finished_jobs': job_history['completed_jobs'],
                    'Training_finished_job_size': np.sum(job_history['completed_jobs_size']),
                    'Training_exceed_deadlines': job_history['exceed_deadline'],
                    'Training_exceed_deadline_job_size': np.sum(job_history['exceed_deadline_size']),
                    'Training Average caching rate: ': Training_finished_jobs/(Training_finished_jobs+Training_exceed_deadlines),
                    'Training Average caching rate(TP): ': Training_finished_job_size/(Training_finished_job_size+Training_exceed_deadline_job_size)}
            wandb.log(metrics)
        # Reset the logs for validation
        print('-------------- Print Training Results---------------------------------------')
        print('Finished jobs: ', Training_finished_jobs/num_eps)
        print('Lost jobs (deadline): ', Training_exceed_deadlines/num_eps)
        print('Average caching rate: ', Training_finished_jobs/(Training_finished_jobs+Training_exceed_deadlines))
        print('Finished job size: ', Training_finished_job_size/num_eps)
        print('Lost job size(deadline)', Training_exceed_deadline_job_size/num_eps)
        print('Average caching rate(TP): ', Training_finished_job_size/(Training_finished_job_size+Training_exceed_deadline_job_size))       
        #for ii in range(n_cpu):
        torch.save(policy_net.state_dict(), os.path.join(wandb.run.dir, "Policy_model.pt"))
        
        rewards_recv = []
        finished_jobs = 0.0
        finished_job_size = 0.0
        lost_jobs = 0.0
        exceed_deadlines = 0.0
        lost_job_size = 0.0
        exceed_deadline_job_size = 0.0
        
        num_eps = 6
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
                obs=next_state
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
            
            finished_jobs_log.append(job_history['completed_jobs'])
            finished_job_size_log.append(np.sum(job_history['completed_jobs_size']))
            exceed_deadlines_log.append(job_history['exceed_deadline'])
            exceed_deadline_job_size_log.append(np.sum(job_history['exceed_deadline_size']))
            metrics = {'Validation_finished_jobs': job_history['completed_jobs'],
                    'Validation_finished_job_size': np.sum(job_history['completed_jobs_size']),
                    'Validation_exceed_deadlines': job_history['exceed_deadline'],
                    'Validation_exceed_deadline_job_size': np.sum(job_history['exceed_deadline_size']),
                    'Validation Average caching rate: ': finished_jobs/(finished_jobs+exceed_deadlines),
                    'Validation Average caching rate(TP): ': finished_job_size/(finished_job_size+exceed_deadline_job_size)}
            wandb.log(metrics)
        env.close()
        
        print('Complete')