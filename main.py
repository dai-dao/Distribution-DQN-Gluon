import gym 
import os
from network import DistributionalDDQNPolicy
from params import DistributionalDDQN_Options


command = 'rm -rf data'
os.system(command)
command = 'mkdir data'
os.system(command)


opt = DistributionalDDQN_Options()
env = gym.make(opt.env_name)
policy = DistributionalDDQNPolicy(opt, env)
policy.learn()
