from typing import Dict, Union

from hpo_bench import MLPTabularBenchmark, RandomForestTabularBenchmark, XGBoostTabularBenchmark

import numpy as np


def sample_random_config(bench_cls) -> Dict[str, Union[int, float]]:
    config: Dict[str, Union[int, float]] = {}
    for k, choices in bench_cls.PARAM_CHOICES.items():
        idx = np.random.randint(len(choices))
        config[k] = choices[idx]

    return config


if __name__ == "__main__":
    for bench_cls in [MLPTabularBenchmark, RandomForestTabularBenchmark, XGBoostTabularBenchmark]:
        print(bench_cls)
        for i in range(8):
            bench = bench_cls(seed=0, dataset_id=bench_cls.DATASET_ID_CHOICES[i])
            for _ in range(10):
                config = sample_random_config(bench_cls)
                print(bench(config))
