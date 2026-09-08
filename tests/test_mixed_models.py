import numpy as np
import pandas as pd

from dsb_states.mixed_models import (
    benjamini_hochberg,
    cluster_bootstrap,
    macro_time_model_contract,
)


def test_bh_is_monotone_in_rank() -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.2])
    adjusted = benjamini_hochberg(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= raw)


def test_cluster_bootstrap_resamples_clusters() -> None:
    table = pd.DataFrame({"group": np.repeat(list("abcd"), 5), "value": np.arange(20)})
    result = cluster_bootstrap(
        table,
        cluster_column="group",
        statistic=lambda frame: frame["value"].mean(),
        iterations=200,
    )
    assert result.cluster_count == 4
    assert result.ci_low < result.estimate < result.ci_high


def test_macro_contract_gates_unmapped_hours() -> None:
    table = pd.DataFrame(
        {
            "hour_post_delivery": [1.5, np.nan],
            "acquisition_id": ["a", "b"],
            "fov_id": ["a", "b"],
            "bundle_id": ["x", "y"],
        }
    )
    assert macro_time_model_contract(table)["status"] == "gated"
