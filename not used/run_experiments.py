# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 02:17:06 2024

@author: ege-demir
"""

import subprocess
import json
from itertools import product
from multiprocessing import Process
from pathlib import Path
import os
import time

# Define parameters to iterate
EXPERIMENTS = {
    "image_count": [15000],
    "exc_neuron_num": [100, 400],
    "tr_label_pred": [True],
    "label_predict_range": [10000],
    "size_selected": [7, 10],
    "notes": ["size experiments 7,10 for 100 400 15k"]
}

# Option to redirect output to log files or suppress it entirely
LOG_OUTPUT = True  # Set to False to suppress output completely
DELAY_BETWEEN_STARTS = 10  # Delay in seconds between starting parallel processes


def generate_experiments(base_config):
    """Generate all possible combinations of parameters to iterate."""
    keys, values = zip(*base_config.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))


def generate_log_file_name(config):
    """Generate a unique log file name based on the configuration parameters."""
    # Concatenate parameter names and values into a string for the file name
    log_name = "_".join([f"{key}={value}" for key, value in config.items()])
    log_name = log_name.replace("/", "_").replace(" ", "_")
    return Path(f"logs/{log_name}_log.txt")


def run_single_experiment(config):
    """Run a single experiment by calling train_parser.py with the given configuration."""
    # Serialize the RF config if it exists
    if "rf_config" in config and config["rf_config"] is not None:
        config["rf_config"] = json.dumps(config["rf_config"])  # Serialize rf_config

    # Build the command for subprocess
    cmd = ["python", "train_parser.py"]
    for key, value in config.items():
        if value is not None:
            # Convert booleans to lowercase strings for the command line
            if isinstance(value, bool):
                value = str(value).lower()
            cmd += [f"--{key}", str(value)]

    # Determine output behavior
    if LOG_OUTPUT:
        log_file = generate_log_file_name(config)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)
    else:
        with open(os.devnull, "w") as devnull:
            subprocess.run(cmd, stdout=devnull, stderr=devnull, check=True)


def run_with_delayed_starts(experiment_configs):
    """Start parallel processes with a delay between each start."""
    processes = []
    for i, config in enumerate(experiment_configs):
        # Create a process for each configuration
        process = Process(target=run_single_experiment, args=(config,))
        processes.append(process)
        process.start()
        print(f"[INFO] Started process {i + 1}/{len(experiment_configs)} with config: {config}")
        time.sleep(DELAY_BETWEEN_STARTS)  # Delay before starting the next process

    # Wait for all processes to finish
    for process in processes:
        process.join()
        print(f"[INFO] Process {process.pid} completed.")


if __name__ == "__main__":
    # Generate experiments
    experiment_configs = list(generate_experiments(EXPERIMENTS))

    # Run experiments with delayed starts
    run_with_delayed_starts(experiment_configs)
