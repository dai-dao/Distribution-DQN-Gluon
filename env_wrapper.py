import mxnet as mx 
from mxnet import nd 
import numpy as np 


def preprocess(opt, raw_frame, currentState = None, initial_state = False):
    raw_frame = nd.array(raw_frame, mx.cpu())
    raw_frame = nd.reshape(nd.mean(raw_frame, axis = 2),shape = (raw_frame.shape[0],raw_frame.shape[1],1))
    raw_frame = mx.image.imresize(raw_frame,  opt.image_size, opt.image_size)
    raw_frame = nd.transpose(raw_frame, (2,0,1))
    raw_frame = raw_frame.astype(np.float32) / 255.
    if initial_state == True:
        state = raw_frame
        for _ in range(opt.frame_len-1):
            state = nd.concat(state , raw_frame, dim = 0)
    else:
        state = mx.nd.concat(currentState[1:,:,:], raw_frame, dim = 0)
    return state

def rew_clipper(rew):
    if rew>0.:
        return 1.
    elif rew<0.:
        return -1.
    else:
        return 0

def step(opt, env, action):
    # Skip frame
    rew = 0
    for skip in range(opt.skip_frame-1):
        next_frame, reward, done,_ = env.step(action)
        rew += reward
        for internal_skip in range(opt.internal_skip_frame-1):
            _ , reward, done,_ = env.step(action)
            rew += reward
            
    next_frame_new, reward, done, _ = env.step(action)
    rew += reward
    
    # Reward clipping
    reward = rew_clipper(rew)
    next_frame = np.maximum(next_frame_new, next_frame)
    
    return next_frame, reward, done, _

