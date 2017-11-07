from mxnet import gluon, nd, autograd
import mxnet.ndarray as F
from env_wrapper import *
from utils import * 
import mxnet as mx 
import numpy as np 


class DistributionalDDQNPolicy(object):
    def __init__(self, opt, env):
        self.opt = opt
        self.env = env 
        self.num_action = env.action_space.n
        self.v_min = -10.0
        self.v_max = 200
        self.atoms = 51
        self.z_values = np.linspace(self.v_min, self.v_max, self.atoms)

        DQN = gluon.nn.Sequential()
        with DQN.name_scope():
            #first layer
            DQN.add(gluon.nn.Conv2D(channels=32, kernel_size=8,strides = 4,padding = 0))
            DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
            DQN.add(gluon.nn.Activation('relu'))
            #second layer
            DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=4,strides = 2))
            DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
            DQN.add(gluon.nn.Activation('relu'))
            #tird layer
            DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=3,strides = 1))
            DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
            DQN.add(gluon.nn.Activation('relu'))
            DQN.add(gluon.nn.Flatten())
            #fourth layer
            DQN.add(gluon.nn.Dense(512,activation ='relu'))
            #fifth layer -> No activation to leave un-normalized
            DQN.add(gluon.nn.Dense(self.num_action * self.atoms))
        self.dqn = DQN
        self.dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
        self.trainer = gluon.Trainer(self.dqn.collect_params(),'RMSProp', \
                                {'learning_rate': opt.lr ,'gamma1':opt.gamma1,'gamma2': opt.gamma2,
                                 'epsilon': opt.rms_eps,'centered' : True})
        self.dqn.collect_params().zero_grad()


        Target_DQN = gluon.nn.Sequential()
        with Target_DQN.name_scope():
            #first layer
            Target_DQN.add(gluon.nn.Conv2D(channels=32, kernel_size=8,strides = 4,padding = 0))
            Target_DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
            Target_DQN.add(gluon.nn.Activation('relu'))
            #second layer
            Target_DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=4,strides = 2))
            Target_DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
            Target_DQN.add(gluon.nn.Activation('relu'))
            #tird layer
            Target_DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=3,strides = 1))
            Target_DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
            Target_DQN.add(gluon.nn.Activation('relu'))
            Target_DQN.add(gluon.nn.Flatten())
            #fourth layer
            Target_DQN.add(gluon.nn.Dense(512,activation ='relu'))
            #fifth layer -> No activation to leave un-normalized
            Target_DQN.add(gluon.nn.Dense(self.num_action * self.atoms))
        self.target_dqn = Target_DQN
        self.target_dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
        self.l2loss = gluon.loss.L2Loss(batch_axis=0)
        self.cross_ent_loss = gluon.loss.SoftmaxCrossEntropyLoss(batch_axis=0, sparse_label=False, from_logits=False)


    def _choose_action(self, state):
        data = nd.array(state.reshape([1, self.opt.frame_len, self.opt.image_size, self.opt.image_size]), self.opt.ctx)
        output = self.dqn(data).asnumpy()
        output = np.reshape(output, (output.shape[0], self.num_action, self.atoms))
        q_values = np.dot(output, self.z_values)
        action = int(np.argmax(q_values, axis=1))
        return action


    def _init_training(self):
        self.frame_counter = 0. # Counts the number of steps so far
        self.annealing_count = 0. # Counts the number of annealing steps
        self.epis_count = 0. # Counts the number episodes so far
        self.replay_memory = Replay_Buffer(self.opt.replay_buffer_size) # Initialize the replay buffer
        self.tot_clipped_reward = []
        self.tot_reward = []
        self.frame_count_record = []
        self.moving_average_clipped = 0.
        self.moving_average = 0.
        self.batch_state = nd.empty((self.opt.batch_size, self.opt.frame_len, self.opt.image_size, 
                                self.opt.image_size), self.opt.ctx)
        self.batch_state_next = nd.empty((self.opt.batch_size, self.opt.frame_len, 
                                     self.opt.image_size, self.opt.image_size), self.opt.ctx)
        self.batches = np.arange(self.opt.batch_size)


    def _epsilon_greedy(self, state, t):
        sample = random.random()
        if self.frame_counter > self.opt.replay_start_size:
            self.annealing_count += 1
        if self.frame_counter == self.opt.replay_start_size:
            print('annealing and laerning are started ')
        
        eps = np.maximum(1. - self.annealing_count / self.opt.annealing_end, self.opt.epsilon_min)
        effective_eps = eps
        if t < self.opt.no_op_max:
            effective_eps = 1.
        # epsilon greedy policy
        if sample < effective_eps:
            action = random.randint(0, self.num_action - 1)
        else:
            action = self._choose_action(state)
        return action 

    
    def _update(self):
        # Train
        if self.frame_counter > self.opt.replay_start_size and \
           self.frame_counter % self.opt.learning_frequency == 0:
            batch_reward, batch_action, batch_done = self.replay_memory.sample(self.opt, 
                                                        self.batch_state, self.batch_state_next)
            
            batch_reward, batch_action, batch_done = batch_reward.asnumpy(), \
                                                     batch_action.asnumpy(), batch_done.asnumpy()
            targets_q = self.dqn(self.batch_state_next).asnumpy()
            targets_q = np.reshape(targets_q, (targets_q.shape[0], self.num_action, self.atoms))
            q_values = np.dot(targets_q, self.z_values)
            target_actions = np.argmax(q_values, axis=1).astype('int32')

            value_eval = self.target_dqn(self.batch_state_next).asnumpy()
            value_eval = np.reshape(value_eval, (value_eval.shape[0], self.num_action, self.atoms))

            distributed_q = value_eval[:, target_actions, :]
            m = np.zeros((self.opt.batch_size, self.z_values.size))

            for j in range(self.atoms):
                tzj = np.fmax(np.fmin(batch_reward - batch_done * self.opt.gamma * self.z_values[j], 
                                      self.v_max), self.v_min)
                bj = ((tzj - self.z_values[0]) /  (self.z_values[1] - self.z_values[0]))
                u = np.ceil(bj).astype('int32')
                l = np.floor(bj).astype('int32')

                m[:, l] = m[:, l] + distributed_q[:, target_actions, j] * (u - bj)
                m[:, u] = m[:, u] + distributed_q[:, target_actions, j] * (bj - l)
            
            m = F.softmax(nd.array(m, self.opt.ctx))
            with autograd.record():
                TD_targets = nd.reshape(self.dqn(self.batch_state), 
                                        (self.opt.batch_size, self.num_action, self.atoms))
                TD_targets_action = TD_targets[self.batches, batch_action]
                loss = self.cross_ent_loss(TD_targets_action, m)
                
            loss.backward()
            self.trainer.step(self.opt.batch_size)

            if self.frame_counter % 800 == 0:
                print('Loss is', nd.sum(loss).asscalar())


    def _save_and_load_checkpoint(self, cum_reward_episode_list):
        if self.frame_counter > self.opt.replay_start_size and  \
           self.frame_counter % self.opt.Target_update == 0:
                check_point = self.frame_counter / (self.opt.Target_update *100)
                fdqn = './data/target_%s_%d' % (self.opt.env_name, int(check_point))
                self.dqn.save_params(fdqn)
                self.target_dqn.load_params(fdqn, self.opt.ctx)

                fnam = './data/cum_reward_episode_Distributional_DDQN_%s' % (self.opt.env_name)
                np.save(fnam, cum_reward_episode_list)


    def learn(self):
        self._init_training()
        cum_reward_episode_list = []
        manualSeed = 1
        mx.random.seed(manualSeed)

        while self.frame_counter < self.opt.max_frame:
            cum_reward_episode = 0
            t = 0
            done = False
            next_frame = self.env.reset()
            state = preprocess(self.opt, next_frame, initial_state = True)
            
            while not done:
                previous_state = state
                action = self._epsilon_greedy(state, t)
                next_frame, reward, done, _ = step(self.opt, self.env, action)
                state = preprocess(self.opt, next_frame, state)
                self.replay_memory.push(previous_state, action, state, reward, done)
                self._update()
                
                t += 1
                self.frame_counter += 1
                cum_reward_episode += reward 

                self._save_and_load_checkpoint(cum_reward_episode_list)                

                if done and self.epis_count % 10. == 0. :
                    print('Epoch count: {}, Episode reward: {}, Frame #: {} / {}, Episode Time steps: {}'
                        .format(self.epis_count, cum_reward_episode, self.frame_counter, self.opt.max_frame, t))
            
            cum_reward_episode_list.append(cum_reward_episode)
            self.epis_count += 1
