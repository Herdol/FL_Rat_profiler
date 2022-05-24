import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

# multiprocess environment
if __name__ == "__main__":
    n_cpu = 1
    #env = SubprocVecEnv([lambda: gym.make('gym_dataOffload:dataCache-v0') for i in range(n_cpu)])
    env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)])

    model = PPO2(CustomPolicy, env, verbose=0,tensorboard_log="./logs/ppo2_tensorboard_3/", n_steps = 1000 )

    """
    reward_list = []
    obs = env.reset()
    i = 0
    while i < 500:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_list.append(rewards[0])
        i += 1
    print(sum(reward_list))

    model.learn(total_timesteps=10000000)
    model.save("ppo2_cartpole")

    del model # remove to demonstrate saving and loading
    """
    model = PPO2.load("ppo2_cartpole")

    # Enjoy trained agent
    rewards_recv = []
    for b in range(100):
        reward_list = []
        obs = env.reset()
        i = 0
        while i < 500:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            reward_list.append(rewards[0])
            i += 1
        rewards_recv.append(np.average(reward_list))

