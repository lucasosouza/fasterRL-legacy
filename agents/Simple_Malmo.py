# coding: utf-8

# imports
import gym
import gym.spaces
import gym_minecraft

# modularization
from wrappers import make_env, wrap_env_malmo

def run_experiment(params, log_dir, local_log_path, random_seed=None):

    # create env and add specific conifigurations to Malmo
    env = make_env(params["DEFAULT_ENV_NAME"])
    env.configure(client_pool=[('127.0.0.1', 10000), ('127.0.0.1', 10001)])
    env.configure(allowDiscreteMovement=["move", "turn"]) # , log_level="INFO")
    env.configure(videoResolution=[420,420])
    env.configure(stack_frames=4)
    env = wrap_env_malmo(env)

    while True:
        action = env.action_space.sample()
        new_state, reward, is_done, _ = env.step(action)
        env.render('human')
        if is_done:
            env.reset()

