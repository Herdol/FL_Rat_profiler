import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
from stable_baselines import PPO2


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128, 128],
                                                          vf=[128, 128, 128, 128])],
                                           feature_extraction="mlp")

# expert policy
# input: observation; output: action
def shortest_first_free_RAT(obs):
    n_cpus = len(obs)
    actions = []
    for ii in range(n_cpus):
        ob = obs[ii]
        RAT_status = ob[-2:]
        buffer = np.reshape(ob[30:-2],(5,3))
        cur_RAT = []
        for jj in range(len(RAT_status)):
            if RAT_status[jj] == 1:
                cur_RAT.append(jj)
        cur_RAT_to_choose = np.random.choice(cur_RAT)
        cur_buffer = buffer.copy()
        time_left = []
        for mm in range(len(cur_buffer)):
            time = cur_buffer[mm][2]
            time_left.append(time)
        idx = np.argmin(time_left)
        # random RAT
        action = idx * 2 + cur_RAT_to_choose
        actions.append(action)
    return actions

# multiprocess environment
if __name__ == "__main__":
    # np.random.seed(0) 
    # random_seeds = np.random.randint(500, size = 5)

    # collect expert experience
    n_cpu = 1
    env = SubprocVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0') for i in range(n_cpu)]) 
    # # env = DummyVecEnv([lambda: gym.make('gym_dataCachingCoding:dataCachingCoding-v0')])
    generate_expert_traj(shortest_first_free_RAT, 'shortest_job_first_8', env, n_episodes=100)
    dataset = ExpertDataset(expert_path='shortest_job_first_8.npz', traj_limitation=1, batch_size=128)

    model = PPO2(CustomPolicy, env, verbose=1,tensorboard_log="./logs/ppo2_tensorboard_1_17", n_steps = 5000)
    # # Pretrain the PPO2 model
    model.pretrain(dataset, n_epochs=2000)

    model.save("ppo2_environment_1_17")
    del model # remove to demonstrate saving and loading