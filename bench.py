"""
* All source is availble here:
    https://github.com/automl/HPOBench/blob/d8b45b1eca9a61c63fe79cdfbe509f77d3f5c779/hpobench/util/data_manager.py#L933-L939

* MLP benchmark is available here:
    https://ndownloader.figshare.com/files/30379005
"""
import json
import os
from typing import Any, Dict, Final, Optional, Union

import numpy as np

import pyarrow.parquet as pq  # type: ignore


DATASET_ID_CHOICES: Final = [31, 53, 3917, 9952, 10101, 146818, 146821, 146822]
EPOCHS_CHOICES: Final = [3, 9, 27, 81, 243]

# DO NOT MODIFY params.json as the order of keys matters
CHOICES = json.load(open("params.json"))
SEEDS: Final = CHOICES["seed"]
CHOICES.pop("seed")


def _validate_query(query: Dict[str, Any], config: Dict[str, Union[int, float]]) -> None:
    if len(query["__index_level_0__"]) != 1:
        raise ValueError(f"There must be only one row for config={config}, but got query={query}")

    queried_config = {k: query[k][0] for k in config.keys()}
    if not all(np.isclose(queried_config[k], v, rtol=1e-3) for k, v in config.items()):
        raise ValueError(f"The query must have the identical config as {config}, but got {queried_config}")


class MLPTabularBenchmark:
    def __init__(
        self,
        data_dir_path: str = "tabular_benchmarks/hpo-bench",
        epochs: int = EPOCHS_CHOICES[-1],
        dataset_id: int = DATASET_ID_CHOICES[0],
        seed: Optional[int] = None
    ):
        file_name = f"nn_{dataset_id}_data.parquet.gzip"
        data_path = os.path.join(os.environ["HOME"], data_dir_path, str(dataset_id), file_name)

        db = pq.read_table(data_path, filters=[("iter", "==", epochs)])
        self._db = db.drop(["iter", "subsample"])
        self._rng = np.random.RandomState(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def __call__(self, config: Dict[str, Union[int, float]]) -> Dict[str, float]:
        config["seed"] = SEEDS[self._rng.randint(len(SEEDS))]
        idx = 0
        for k, v in config.items():
            idx = self._db[k].index(v, start=idx).as_py()

        query = self._db.take([idx]).to_pydict()
        _validate_query(query, config)
        return query["result"][0]["info"]["val_scores"]
