"""
An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole. And uses a DQN with
1M steps.
"""
import gym

from garage.envs import normalize
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import GreedyPolicy
from garage.tf.q_functions import DiscreteMLPQFunction

env = TfEnv(normalize(gym.make("CartPole-v0")))

replay_buffer = SimpleReplayBuffer(
    env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

policy = GreedyPolicy(env_spec=env.spec, decay_period=1e6)

qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(8, 4))

algo = DQN(
    env=env,
    policy=policy,
    qf=qf,
    qf_lr=0.001,
    replay_buffer=replay_buffer,
    n_timestamps=int(1e5),
    min_buffer_size=1e5,
    n_train_steps=1,
    target_network_update_freq=10000,
    buffer_batch_size=512,
    dueling=False)

algo.train()
