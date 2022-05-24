import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128, 128],
                                                          vf=[128, 128, 128, 128])],
                                           feature_extraction="mlp")


# multiprocess environment
if __name__ == "__main__":
    np.random.seed(0) 
    random_seeds = np.random.randint(500, size = 5)
    for ii in range(1):
    # for ii in range(0, len(random_seeds)):
        seed = random_seeds[ii]
        
        n_cpu = 12
        env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)])
        #env = gym.make('gym_dataCachingCoding:dataCachingCoding-v0')

        tensorboard_log = "./logs/ppo2_tensorboard_1_18_2"
        model_name = "ppo2_environment_1_18_1"
        model = PPO2(CustomPolicy, env, verbose=0,tensorboard_log=tensorboard_log, n_steps = 5000 )
        #model = PPO2.load(model_name, env=env, verbose=0, tensorboard_log=tensorboard_log, n_steps = 5000)
        print("Model loaded - ", model_name)
        
        # reward_list = []
        # obs = env.reset()
        # i = 0
        # while i < 500:
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones, info = env.step(action)
        #     reward_list.append(rewards[0])
        #     i += 1
        # print(sum(reward_list))
        save_model_name = "ppo2_environment_1_18_2"
        model.learn(total_timesteps=50000)#, seed=seed
        model.save(save_model_name)
        print('Model saved.')

        del model # remove to demonstrate saving and loading
        
        model = PPO2.load(save_model_name, env=env, n_steps=5000)

        # Enjoy trained agent
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
        caching_rate_log=[]
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
            caching_rate_log.append(finished_jobs/(finished_jobs+exceed_deadlines))
            finished_jobs_log.append(job_history[ii]['completed_jobs'])
            finished_job_size_log.append(np.sum(job_history[ii]['completed_jobs_size']))
            exceed_deadlines_log.append(job_history[ii]['exceed_deadline'])
            exceed_deadline_job_size_log.append(np.sum(job_history[ii]['exceed_deadline_size']))
            
        print('-------------- Print Results---------------------------------------')
        print('Finished jobs: ', finished_jobs/num_eps/n_cpu)
        print('Lost jobs (deadline): ', exceed_deadlines/num_eps/n_cpu)
        print('Average caching rate: ', finished_jobs/(finished_jobs+exceed_deadlines))
        print('Finished job size: ', finished_job_size/num_eps/n_cpu*1)
        print('Lost job size(deadline)', exceed_deadline_job_size/num_eps/n_cpu*1)
        print('Average caching rate(TP): ', finished_job_size/(finished_job_size+exceed_deadline_job_size))
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(finished_jobs_log)
        axs[0, 0].set_title('finished_jobs')
        axs[0, 1].plot(finished_job_size_log, 'tab:orange')
        axs[0, 1].set_title('finished_job_size_log')
        axs[1, 0].plot(exceed_deadlines_log, 'tab:green')
        axs[1, 0].set_title('exceed_deadlines_log')
        axs[1, 1].plot(caching_rate_log, 'tab:red')
        axs[1, 1].set_title('Average caching rate')
        print("plotting is done")