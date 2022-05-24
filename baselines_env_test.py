import numpy as np
# import tensorflow as tf
import pickle
import gym

env = gym.make('gym_dataCachingCoding:dataCachingCoding-v0')

# shortest job first: to always choose the job with minimum time left
# random RAT: put the chosen job to a random RAT
def shortest_first_rand_RAT(validate_episodes, max_steps):
    rewards_list = []
    finished_jobs = []
    finished_size = []
    exceed_deadlines = []
    exceed_deadline_size = []
    reward_per_step = []
    for ep in range(validate_episodes):
        total_reward = 0
        t = 0
        env.reset(USER_USAGE = False)
        while env.bs['init'].time < max_steps:
            cur_buffer = env.bs['init'].buffer.buffer.copy()
            time_left = []
            for ii in range(len(cur_buffer)):
                time = cur_buffer[ii][2]
                time_left.append(time)
            idx = np.argmin(time_left)
            # random RAT
            action = idx * 2 + np.random.randint(2)
            _, r, _, _ = env.step(action)
            total_reward += r
            reward_per_step.append(r)
            t += 1        
        rewards_list.append(total_reward)
        
        if ep % 1 == 0:
            avg_score = np.mean(rewards_list[-24:])
            print("Episode: " + str(ep) + " Ave Reward: " + str(avg_score))
            print('Reward per step', np.mean(reward_per_step))

    # record finished jobs and lost jobs in this ep
    job_history = env.return_jobs()
    finished_jobs.append(job_history['completed_jobs'])
    finished_size.append(job_history['completed_jobs_size'])
    exceed_deadlines.append(job_history['exceed_deadline'])
    exceed_deadline_size.append(job_history['exceed_deadline_size'])

    with open('finished_jobs_bl.txt', 'wb') as fp:
        pickle.dump(finished_jobs, fp)
    with open('finished_jobs_size_bl.txt', 'wb') as fp:
        pickle.dump(finished_size, fp)
    with open('exceed_deadlines_bl.txt', 'wb') as fp:
        pickle.dump(exceed_deadlines, fp)
    with open('exceed_deadline_size_bl.txt', 'wb') as fp:
        pickle.dump(exceed_deadline_size, fp)
    with open('rewards_list_bl.txt', 'wb') as fp:
        pickle.dump(rewards_list, fp)
    print('Reward per step final', np.mean(reward_per_step))

# put the chosen shortest job to a free RAT
def shortest_first_free_RAT(validate_episodes, max_steps):
    rewards_list = []
    finished_jobs = []
    finished_size = []
    exceed_deadlines = []
    exceed_deadline_size = []
    reward_per_step = []
    for ep in range(validate_episodes):
        total_reward = 0
        t = 0
        env.reset(USER_USAGE = False)
        while env.bs['init'].time < max_steps:
            cur_RAT = []
            for jj in range(len(env.bs['init'].RAT_status())):
                if env.bs['init'].RAT_status()[jj] == 1:
                    cur_RAT.append(jj)
            cur_RAT_to_choose = np.random.choice(cur_RAT)
            cur_buffer = env.bs['init'].buffer.buffer.copy()
            time_left = []
            for ii in range(len(cur_buffer)):
                time = cur_buffer[ii][2]
                time_left.append(time)
            idx = np.argmin(time_left)
            # random RAT
            action = idx * 2 + cur_RAT_to_choose
            _, r, _, _ = env.step(action)
            total_reward += r
            t += 1
            reward_per_step.append(r)
        
        rewards_list.append(total_reward)
        
        if ep % 1 == 0:
            avg_score = np.mean(rewards_list[-24:])
            print("Episode: " + str(ep) + " Ave Reward: " + str(avg_score))
            print('Reward per step', np.mean(reward_per_step))

    # record finished jobs and lost jobs in this ep
    job_history = env.return_jobs()
    finished_jobs.append(job_history['completed_jobs'])
    finished_size.append(job_history['completed_jobs_size'])
    exceed_deadlines.append(job_history['exceed_deadline'])
    exceed_deadline_size.append(job_history['exceed_deadline_size'])

    with open('finished_jobs_bl.txt', 'wb') as fp:
        pickle.dump(finished_jobs, fp)
    with open('finished_jobs_size_bl.txt', 'wb') as fp:
        pickle.dump(finished_size, fp)
    with open('exceed_deadlines_bl.txt', 'wb') as fp:
        pickle.dump(exceed_deadlines, fp)
    with open('exceed_deadline_size_bl.txt', 'wb') as fp:
        pickle.dump(exceed_deadline_size, fp)
    with open('rewards_list_bl.txt', 'wb') as fp:
        pickle.dump(rewards_list, fp)
    print('Reward per step final', np.mean(reward_per_step))

# always choose the vehicle which can get the highest data rate for both RATs
# if there're more than one jobs asssigned for the chosen vehicle, choose the shortest one
def nearest_vehicle_policy(validate_episodes, max_steps, num_actions):
    rewards_list = []
    finished_jobs = []
    finished_size = []
    exceed_deadlines = []
    exceed_deadline_size = []
    reward_per_step = []
    for ep in range(validate_episodes):
        total_reward = 0
        t = 0
        env.reset(USER_USAGE = False)
        while env.bs['init'].time < max_steps:
            distances = env.calculate_distances()
            cur_buffer_v = np.array(env.bs['init'].buffer.buffer.copy())[:,1]
            # if lte is free
            if env.bs['init'].RAT_status()[0] == 1:
                cur_distances = distances['lte']
                cur_distances_sort = cur_distances.copy()
                cur_distances_sort.sort()
                for ii in range(len(cur_distances)):
                    v_to_choose = np.where(cur_distances==cur_distances_sort[ii])[0]
                    # find if there's a job for this vehicle
                    if v_to_choose in cur_buffer_v:
                        idx_job = np.where(cur_buffer_v==v_to_choose)
                        job_time_left = np.array(np.array(env.bs['init'].buffer.buffer)[idx_job,2])
                        job_idx = idx_job[0][np.argmin(job_time_left[0])]
                        action = job_idx * 2
                        break
                    else:
                        action = num_actions - 1
            # if lte is not free and mmWaves is free:
            if env.bs['init'].RAT_status()[0] == 0 and env.bs['init'].RAT_status()[1] == 1:
                cur_distances = distances['mmWaves']
                cur_distances_sort = cur_distances.copy()
                cur_distances_sort.sort()
                for ii in range(len(cur_distances)):
                    v_to_choose = np.where(cur_distances==cur_distances_sort[ii])[0]
                    # find if there's a job for this vehicle
                    if v_to_choose in cur_buffer_v:
                        idx_job = np.where(cur_buffer_v==v_to_choose)
                        job_time_left = np.array(np.array(env.bs['init'].buffer.buffer)[idx_job,2])
                        job_idx = idx_job[0][np.argmin(job_time_left[0])]
                        action = job_idx * 2 + 1
                        break
                    else:
                        action = num_actions - 1
            _, r, _, _ = env.step(action)
            total_reward += r
            reward_per_step.append(r)
            t += 1
        
        rewards_list.append(total_reward)
        
        if ep % 1 == 0:
            avg_score = np.mean(rewards_list[-24:])
            print("Episode: " + str(ep) + " Ave Reward: " + str(avg_score))
            print('Reward per step', np.mean(reward_per_step))

    # record finished jobs and lost jobs in this ep
    job_history = env.return_jobs()
    finished_jobs.append(job_history['completed_jobs'])
    finished_size.append(job_history['completed_jobs_size'])
    exceed_deadlines.append(job_history['exceed_deadline'])
    exceed_deadline_size.append(job_history['exceed_deadline_size'])

    with open('finished_jobs_bl.txt', 'wb') as fp:
        pickle.dump(finished_jobs, fp)
    with open('finished_jobs_size_bl.txt', 'wb') as fp:
        pickle.dump(finished_size, fp)
    with open('exceed_deadlines_bl.txt', 'wb') as fp:
        pickle.dump(exceed_deadlines, fp)
    with open('exceed_deadline_size_bl.txt', 'wb') as fp:
        pickle.dump(exceed_deadline_size, fp)
    with open('rewards_list_bl.txt', 'wb') as fp:
        pickle.dump(rewards_list, fp)
    print('Reward per step final', np.mean(reward_per_step))



# shortest_first_rand_RAT(validate_episodes=20, max_steps=125)
shortest_first_free_RAT(validate_episodes=20, max_steps=125)
# nearest_vehicle_policy(validate_episodes=20, max_steps=125, num_actions=11)
