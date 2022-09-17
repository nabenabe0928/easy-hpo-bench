import json
from typing import Final, Optional

from hpo_bench.base_bench import BaseTabularBenchmark
from hpo_bench.constants import DATASET_ID_PATH, DEFAULT_DATA_DIR_NAME, PARAMS_PATH


ALGO_NAME = "mlp"
EPOCHS_CHOICES: Final = [3, 9, 27, 81, 243]


class MLPTabularBenchmark(BaseTabularBenchmark):
    DATASET_ID_CHOICES: Final = json.load(open(DATASET_ID_PATH))[ALGO_NAME]
    PARAM_CHOICES: Final = json.load(open(PARAMS_PATH))[ALGO_NAME]

    def __init__(
        self,
        data_dir_path: str = DEFAULT_DATA_DIR_NAME,
        epochs: int = EPOCHS_CHOICES[-1],
        dataset_id: int = DATASET_ID_CHOICES[0],
        seed: Optional[int] = None,
    ):
        super().__init__(
            algo_name="nn",
            data_dir_path=data_dir_path,
            budget=epochs,
            budget_name="iter",
            dataset_id=dataset_id,
            seed=seed
        )
