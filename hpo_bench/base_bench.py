import os
from typing import Dict, Final, Optional, Union

from hpo_bench.utils import _validate_query

import numpy as np

import pyarrow.parquet as pq  # type: ignore


SEEDS: Final = [665, 1319, 7222, 7541, 8916]


class BaseTabularBenchmark:
    def __init__(
        self,
        algo_name: str,
        data_dir_path: str,
        budget: int,
        budget_name: str,
        dataset_id: int,
        seed: Optional[int],
    ):
        file_name = f"{algo_name}_{dataset_id}_data.parquet.gzip"
        data_path = os.path.join(os.environ["HOME"], data_dir_path, str(dataset_id), file_name)

        db = pq.read_table(data_path, filters=[(budget_name, "==", budget)])
        self._db = db.drop([budget_name, "subsample"])
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
