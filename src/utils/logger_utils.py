import json
import os
import time
from collections import defaultdict

import numpy as np

from src.tiny_agent.models import TINY_AGENT_DIR

# Global variable to toggle logging
LOG_ENABLED = True
LOG_TO_FILE = False
LOG_FILE_PATH = os.path.join(TINY_AGENT_DIR, "log.txt")

# Create the log file if it doesn't exist
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w") as f:
        pass


class Logger:
    def __init__(self) -> None:
        self._latency_dict = defaultdict(list)
        self._answer_dict = defaultdict(list)
        self._label_dict = defaultdict(list)

    def log(self, latency: float, answer: str, label: str, key: str) -> None:
        self._latency_dict[key].append(latency)
        self._answer_dict[key].append(answer)
        self._label_dict[key].append(label)

    def _get_mean_latency(self, key: str) -> float:
        latency_array = np.array(self._latency_dict[key])
        return latency_array.mean(), latency_array.std()

    def _get_accuracy(self, key: str) -> float:
        answer_array = np.array(self._answer_dict[key])
        label_array = np.array(self._label_dict[key])
        return (answer_array == label_array).mean()

    def get_results(self, key: str) -> dict:
        mean_latency, std_latency = self._get_mean_latency(key)
        accuracy = self._get_accuracy(key)
        return {
            "mean_latency": mean_latency,
            "std_latency": std_latency,
            "accuracy": accuracy,
        }

    def save_result(self, key: str, path: str):
        with open(f"{path}/dev_react_results.csv", "w") as f:
            for i in range(len(self._answer_dict[key])):
                f.write(f"{self._answer_dict[key][i]},{self._latency_dict[key][i]}\n")


def get_logger() -> Logger:
    return Logger()


# Custom print function to toggle logging


def enable_logging(enable=True):
    """Toggle logging on or off based on the given argument."""
    global LOG_ENABLED
    LOG_ENABLED = enable


def enable_logging_to_file(enable=True):
    """Toggle logging on or off based on the given argument."""
    global LOG_TO_FILE
    LOG_TO_FILE = enable


def log(*args, block=False, **kwargs):
    """Print the given string only if logging is enabled."""
    if LOG_ENABLED:
        if block:
            print("=" * 80)
        print(*args, **kwargs)
        if block:
            print("=" * 80)
    if LOG_TO_FILE:
        with open(LOG_FILE_PATH, "a") as f:
            if block:
                print("=" * 80, file=f)
            print(*args, **kwargs, file=f)
            if block:
                print("=" * 80, file=f)


def flush_results(save_path, results):
    print("Saving results")
    json.dump(results, open(save_path, "w"), indent=4)
