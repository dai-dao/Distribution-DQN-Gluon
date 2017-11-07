from collections import namedtuple
from mxnet import nd
import numpy as np 
import random


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))
class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0
    
    
    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    
    
    def sample(self, opt, batch_state, batch_state_next):
        transitions = random.sample(self.memory, opt.batch_size)
        batch = Transition(*zip(*transitions))
        
        for j in range(opt.batch_size):
            batch_state[j] = nd.array(batch.state[j],opt.ctx)
            batch_state_next[j] = nd.array(batch.next_state[j],opt.ctx)

        batch_reward = nd.array(batch.reward, opt.ctx)
        batch_action = nd.array(batch.action, opt.ctx)
        batch_done = nd.array(batch.done, opt.ctx)

        return batch_reward, batch_action, batch_done
