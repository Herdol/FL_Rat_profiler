"""
This shows a brief introduction to how to run the environment.

Quick note:

actions_todo which is in the env.step() method call is non-standard. It allows for the following occurance
- If there are two RATs available, we may want to select two jobs for transmission at the same time.
- DQN doesn't really natively support this, so I've adapted it to do this.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('gym_dataOffload:dataCache-v0') #this is the import
env.reset(USER_USAGE = False)
state = env.state()

#### Define some number of iterations for the episode ####
i_max = 1000
rewards_list = []
#### This is the main loop which is where we select actions ####
for ep in range(100):
    total_reward = 0.0
    i = 0
    while i < i_max:
        actions_todo = np.where(env.bs['init'].RAT_status() == 1.0)[0].shape[0] #count number of RATs available
        # print("There are %s RATs available"%actions_todo)
        for j in range(actions_todo): # allow for the selection of job for whichever rats are available.
            # print('RAT status', env.bs['init'].RATs)
            action = env.action_space.sample() #select random job
            # if action != 20: #if not do nothing. 20 is do nothing
            #     index, RAT = np.divmod(action,2) #this is for the purpose of the print statement on the nxt line
            #     print("Action selected was %i = job %s for RAT %s\nThis results in job\n %s\n"%(action, index, RAT,env.bs['init'].buffer.buffer[index]))
            # else:
            #     print("Do nothing selected")
            # Next line is how actions are sent to env. actions_todo is non-standard, allows for 2 RATs to be set
            next_state, reward, info, done = env.step(action)
            total_reward += reward
            # print(total_reward)
            # print('RAT status', env.bs['init'].RATs)
            # print('reward', reward)
            state = next_state
            # print('v pos', env.vehicle_positions()[:,0:2])
            # print('buffer', env.bs['init'].buffer.buffer)
            # print('-------------------------------------------------------------------------')
        else:
            next_state = env.update_env()
            state = next_state
            i += 1 #increment steps
    env.reset(USER_USAGE = False)
    state = env.state()
    rewards_list.append(total_reward)
    print(ep, ': ', total_reward, 't: ', i)
    # print(env.bs['init'].RATs)
plt.plot(rewards_list)
print('Ave Rewards in 200 eps', np.mean(rewards_list))
