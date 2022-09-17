"""
* All source is availble here:
    https://github.com/automl/HPOBench/blob/d8b45b1eca9a61c63fe79cdfbe509f77d3f5c779/hpobench/util/data_manager.py#L933-L939

* MLP benchmark is available here:
    https://ndownloader.figshare.com/files/30379005

* XGBoost benchmark is available here:
    https://ndownloader.figshare.com/files/30469920

* Random Forest benchmark is available here:
    https://ndownloader.figshare.com/files/30469089
"""
from hpo_bench.mlp import MLPTabularBenchmark
from hpo_bench.rf import RandomForestTabularBenchmark
from hpo_bench.xgb import XGBoostTabularBenchmark


__all__ = [
    "MLPTabularBenchmark",
    "RandomForestTabularBenchmark",
    "XGBoostTabularBenchmark",
]
