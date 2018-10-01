# coding: utf-8

import os
import json
import argparse
from itertools import product
from datetime import datetime
import numpy as np
from utils import *

if __name__ == "__main__":

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

    for params in experiments:

        # create an ID for the experiment
        now = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
        experiment_id = "-".join([params["METHOD"], params["DEFAULT_ENV_NAME"], now])
        runs_dir = os.path.join("runs", experiment_id)

        # dumps json with experiment hyperparameters
        with open("logs/" + experiment_id + ".json", "w") as f:
            json.dump(params, f)

        # need to find a way to encapuslate method as well
        method = methods[params["METHOD"]]
        local_log = method(params, runs_dir)
        local_log_path = "results/" + experiment_id + '.json'

        # output json
        with open( local_log_path , "w") as f:
            json.dump(local_log, f)

        print("Experiment complet. Results found at: " + local_log_path)




