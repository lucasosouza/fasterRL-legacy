# coding: utf-8

import gym

def make_env(env_name):
    """ Currently no transformations are required """
    
    env = gym.make(env_name)    
    return env

