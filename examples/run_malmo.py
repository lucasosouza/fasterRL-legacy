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


"""
Estrutura do MinecraftBasic
Reward -1 for sending command
Reward -10000 for death
Reward 1000 for find goal (goal is gold, diamond or redstone block)
Reward 1000 for out of time
Reward 2- tpr touching gold, diamond or redstone ore
https://github.com/tambetm/gym-minecraft/blob/master/gym_minecraft/assets/basic.xml
"""

def gen_experiments(params_file):

    with open(params_file, 'rb') as f:
        params = json.load(f)

        # create array of experiments
        experiments = []
        search_method = params["SEARCH_METHOD"]

        # if hypothesis, do hypothesis testing with one dependent variable for each test
        if search_method == "hypothesis":

            # define default experiment
            default_experiment = {}
            for param, value in params.items():
                if type(value) != list:
                    default_experiment[param] = value
                else:
                    default_experiment[param] = value[0]
            experiments.append(default_experiment)

            # iterate again and create additional hypothesis testing experiments
            for param, value in params.items():
                if type(value) == list:
                    for val in value[1:]:
                        new_experiment = default_experiment.copy()
                        new_experiment[param] = val
                        experiments.append(new_experiment)

        # else, generate the combination in a grid or grid like search approach
        else:

            # get right list of parameters according to search method
            for param, value in params.items():
                if type(value) != list:
                    params[param] = [(param, value)]
                else:
                    if search_method == "grid":
                        # values: values to try
                        params[param] = [(param, val) for val in value]             
                    elif search_method == "random":
                        # values: lower_bound, upper_bound, n_vals
                        params[param] = [(param, val) for val in np.random.uniform(*value)]
                    elif search_method == "gaussian_random":
                        # values: mean, standard deviation, n_vals
                        params[param] = [(param, val) for val in np.random.normal(*value)]
                    else:
                        raise ValueError("Method search {} not a valid option.".format(search_method))                    

            combinations = product(*params.values())
            for c in combinations:
                experiments.append(dict(c))

    return experiments

if __name__ == "__main__":

	# identify log directory
    if "LOG_DIR" in os.environ:
        log_root = os.environ["LOG_DIR"]
    else:
        log_root = "./"

    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("params") # mandatory

    # load parameters
    params_file = parser.parse_args().params
    experiments = gen_experiments(params_file)
    print("Number of experiments to be conducted: ", str(len(experiments)))

    # define variables for multiprocessing
    # NUM_PROCESSES = 1
    # pool = Pool(processes=NUM_PROCESSES)
    # processes = []

    ##### run experiment
    ####################
    for params in experiments:

        # create an ID for the experiment
        now = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")[:-4]
        sleep(0.01) # avoids having file with exact name
        experiment_id = "-".join([params["METHOD"], params["DEFAULT_ENV_NAME"], now])
        print("running: ", experiment_id)
        runs_dir = os.path.join(log_root, "runs", experiment_id)

        # dumps json with experiment hyperparameters
        with open(os.path.join(log_root, "logs", experiment_id + ".json"), "w") as f:
            json.dump(params, f)

        RANDOM_SEED = 42

        method = methods[params["METHOD"]]
        local_log_path = os.path.join(log_root, "results", experiment_id + '.json')

        # run method
        method(params, runs_dir, local_log_path)

    #     # run methods in parallel
    #     p = pool.apply_async(method, (params, runs_dir, local_log_path, RANDOM_SEED))
    #     processes.append(p)

    # for p in processes:
    #     p.get()




