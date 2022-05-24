# model validation
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
from stable_baselines import PPO2
from stable_baselines.a2c.utils import total_episode_reward_logger

if __name__ == "__main__":
    n_cpu = 12
    env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)]) 
    model = PPO2.load("ppo2_environment_1_16_1", env=env, n_steps=5000)

    # Enjoy trained agent
    rewards_recv = []
    finished_jobs = 0.0
    finished_job_size = 0.0
    lost_jobs = 0.0
    exceed_deadlines = 0.0
    lost_job_size = 0.0
    exceed_deadline_job_size = 0.0
    num_eps = 20
    for b in range(num_eps):
        print('ep', b)
        reward_list = []
        obs = env.reset()
        time_ep = 0.0
        while time_ep < 125:
            start_time = env.get_attr('time')[0]
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            reward_list.append(rewards[0])
            end_time = env.get_attr('time')[0]
            time_elapsed = end_time - start_time
            time_ep += max(time_elapsed, 0)
        rewards_recv.append(np.average(reward_list))
    job_history = env.get_attr('history')

    for ii in range(n_cpu):
        finished_jobs += job_history[ii]['completed_jobs']
        finished_job_size += np.sum(job_history[ii]['completed_jobs_size'])
        exceed_deadlines += job_history[ii]['exceed_deadline']
        exceed_deadline_job_size += np.sum(job_history[ii]['exceed_deadline_size'])
    print('-------------- Print Results---------------------------------------')
    print('Finished jobs: ', finished_jobs/num_eps/n_cpu)
    print('Lost jobs (deadline): ', exceed_deadlines/num_eps/n_cpu)
    print('Average caching rate(job): ', finished_jobs/(finished_jobs+exceed_deadlines))
    print('Finished job size: ', finished_job_size/num_eps/n_cpu*2 )
    print('Lost job size(deadline)', exceed_deadline_job_size/num_eps/n_cpu*2)
    print('Average caching rate(TP): ', finished_job_size/(finished_job_size+exceed_deadline_job_size))


if __name__ == "__main__":
    n_cpu = 1
    env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)]) 
    model = PPO2.load("ppo2_environment_1_16_1", env=env, n_steps=5000)

    # Enjoy trained agent
    rewards_recv = []
    finished_jobs = 0.0
    finished_job_size = 0.0
    lost_jobs = 0.0
    exceed_deadlines = 0.0
    lost_job_size = 0.0
    exceed_deadline_job_size = 0.0
    num_eps = 1
    for b in range(num_eps):
        print('ep', b)
        reward_list = []
        obs = env.reset()
        time_ep = 0.0
        done = False
        t = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            reward_list.append(rewards[0])
            t += 1
        print(t)
        rewards_recv.append(np.sum(reward_list))

    print('-------------- Print Results---------------------------------------')
    print('episode reward', np.mean(rewards_recv))
