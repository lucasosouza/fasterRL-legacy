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

if __name__ == "__main__":

    if "LOG_DIR" in os.environ:
        log_root = os.environ["LOG_DIR"]
    else:
        log_root = "./"

    # load paramters
    parser = argparse.ArgumentParser()
    parser.add_argument("params") # mandatory

    params_file = parser.parse_args().params
    with open(params_file, 'rb') as f:
        params = json.load(f)

    # get right list of parameters according to search method
        search_method = params["SEARCH_METHOD"]
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

    # generate the combination
    experiments = []
    combinations = product(*params.values())
    for c in combinations:
        experiments.append(dict(c)) 

    NUM_PROCESSES = cpu_count()
    pool = Pool(processes=NUM_PROCESSES)
    processes = []
    for params in experiments:

        # create an ID for the experiment
        now = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")[:-4]
        sleep(0.01)
        experiment_id = "-".join([params["METHOD"], params["DEFAULT_ENV_NAME"], now])
        print("running: ", experiment_id)
        runs_dir = os.path.join(log_root, "runs", experiment_id)

        # dumps json with experiment hyperparameters
        with open(os.path.join(log_root, "logs", experiment_id + ".json"), "w") as f:
            json.dump(params, f)

        RANDOM_SEED = 42

       # need to find a way to encapsulate method as well
        # pprint(params)
        method = methods[params["METHOD"]]
        local_log_path = os.path.join(log_root, "results", experiment_id + '.json')

        # run methods in parallel
        p = pool.apply_async(method, (params, runs_dir, local_log_path, RANDOM_SEED))
        processes.append(p)

    for p in processes:
        p.get()




