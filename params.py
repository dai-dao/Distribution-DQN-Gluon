import mxnet as mx 

class DistributionalDDQN_Options:
    def __init__(self):
        #Architecture
        self.batch_size = 32 # The size of the batch to learn the Q-function
        self.image_size = 84 # Resize the raw input frame to square frame of size 80 by 80 
        #Trickes
        self.replay_buffer_size = 50000 # The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
        self.learning_frequency = 100 # With Freq of 1/4 step update the Q-network
        self.skip_frame = 4 # Skip 4-1 raw frames between steps
        self.internal_skip_frame = 4 # Skip 4-1 raw frames between skipped frames
        self.frame_len = 4 # Each state is formed as a concatination 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
        self.Target_update = 3000 # Update the target network each 10000 steps
        self.epsilon_min = 0.1 # Minimum level of stochasticity of policy (epsilon)-greedy
        self.annealing_end = 1000000. # The number of step it take to linearly anneal the epsilon to it min value
        self.gamma = 0.99 # The discount factor
        self.replay_start_size = 500 # Start to backpropagated through the network, learning starts
        self.no_op_max = 30 / self.skip_frame # Run uniform policy for first 30 times step of the beginning of the game
        
        #otimization
        self.num_episode = 10000000 # Number episode to run the algorithm
        self.max_frame = 200000000
        self.lr = 0.00025 # RMSprop learning rate
        self.gamma1 = 0.95 # RMSprop gamma1
        self.gamma2 = 0.95 # RMSprop gamma2
        self.rms_eps = 0.01 # RMSprop epsilon bias
        self.ctx = mx.cpu() # Enables gpu if available, if not, set it to mx.cpu()
        self.env_name = 'AssaultNoFrameskip-v4'