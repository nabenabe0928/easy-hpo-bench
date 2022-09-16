from typing import Dict, Union

from bench import CHOICES, DATASET_ID_CHOICES, MLPTabularBenchmark

import numpy as np


def sample_random_config() -> Dict[str, Union[int, float]]:
    config: Dict[str, Union[int, float]] = {}
    for k, choices in CHOICES.items():
        idx = np.random.randint(len(choices))
        config[k] = choices[idx]

    return config


if __name__ == "__main__":
    for i in range(8):
        bench = MLPTabularBenchmark(seed=0, dataset_id=DATASET_ID_CHOICES[i])
        for _ in range(100):
            config = sample_random_config()
            print(bench(config))
