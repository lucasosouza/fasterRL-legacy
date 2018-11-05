# coding: utf-8

import os
import json
import argparse
from itertools import product
from datetime import datetime
import numpy as np
from utils import *
from pprint import pprint
from time import sleep

from multiprocessing import Pool, cpu_count

# added
from wrappers import wrap_env_marlo
from agents import DQN, Agent

def get_join_tokens(env_name):
    if marlo.is_grading():
        """
            In the crowdAI Evaluation environment obtain the join_tokens 
            from the evaluator
            
            the `params` parameter passed to the `evaluator_join_token` only allows
            the following keys : 
                    "seed",
                    "tick_length",
                    "max_retries",
                    "retry_sleep",
                    "step_sleep",
                    "skip_steps",
                    "videoResolution",
                    "continuous_to_discrete",
                    "allowContinuousMovement",
                    "allowDiscreteMovement",
                    "allowAbsoluteMovement",
                    "add_noop_command",
                    "comp_all_commands"
                    # TODO: Add this to the official documentation ? 
                    # Help Wanted :D Pull Requests welcome :D 
        """
        join_tokens = marlo.evaluator_join_token(params={})

    else:
        """
            When debugging locally,
            Please ensure that you have a Minecraft client running on port 10000
            by doing : 
            $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
        """
        client_pool = [('127.0.0.1', 10000)]
        join_tokens = marlo.make(env_name,
                                 params={
                                    "client_pool": client_pool
                                 })
    return join_tokens


def init_environment(params):

	join_tokens = get_join_tokens(params["ENV_NAME"])
    # As this is a single agent scenario,there will just be a single token
    assert len(join_tokens) == 1
    join_token = join_tokens[0]

    # initialize environment    
    env = marlo.init(join_token)
    env = wrap_env_marlo(env)

   return env

def create_agent(params, agent_id, env):

    # initialize networks and load weights
    net = DQN(env.observation_space.shape, env.action_space.n, params["DEVICE"]).to(device)
    net.load_state_dict(torch.load(params["WEIGHTS"]))

    # initialize agent
    agent = Agent(agent_id, env, net, params)

    return agent

def run_episode(params, agent):
    """
    Single episode run
    """

	# recreate and start environment
	env = init_environment(params["ENV_NAME"])
	agent.state = env.reset()

	# loop
    episode_over = False
    while not episode_over:
		episode_over, done_reward = agent.play_step(device=device)

	env.close()

if __name__ == "__main__":

    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("params") # mandatory

    # load parameters
    params_file = parser.parse_args().params
    method = methods[params["METHOD"]]

	# create a sample environment to be used to define agent params
	env = init_environment(params["ENV_NAME"])
	agent = create_agent(params, "agent0", env)

    """
        In case of debugging locally, run the episode just once
        and in case of when the agent is being evaluated, continue 
        running episodes for as long as the evaluator keeps supplying
        join_tokens.
    """
    if not marlo.is_grading():
        print("Running single episode...")
        run_episode(params, agent)
    else:
        while True:
            run_episode(params, agent)


