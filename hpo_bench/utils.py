from typing import Any, Dict, Union

import numpy as np


def _validate_query(
    query: Dict[str, Any], config: Dict[str, Union[int, float]]
) -> None:
    if len(query["__index_level_0__"]) != 1:
        raise ValueError(
            f"There must be only one row for config={config}, but got query={query}"
        )

    queried_config = {k: query[k][0] for k in config.keys()}
    if not all(np.isclose(queried_config[k], v, rtol=1e-3) for k, v in config.items()):
        raise ValueError(
            f"The query must have the identical config as {config}, but got {queried_config}"
        )
