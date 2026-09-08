from __future__ import annotations

import pandas as pd

from dsb_states.v521_case_selection import CompletePairCaseSelection
from dsb_states.v521_case_selection_frame_exclusion import (
    exclude_acquisition_frame_counts_and_rerank,
)


def test_exact_t5_t10_exclusion_and_reranking() -> None:
    summary = pd.DataFrame(
        {
            "bundle_id": ["t5", "t10", "kept_short", "kept_long"],
            "hour_post_delivery": [3.5, 10.0, 2.0, 4.5],
            "movie_frames": [5, 10, 20, 50],
            "mean_separation_nm": [1.0, 1000.0, 20.0, 500.0],
            "global_shortest_rank": [1, 4, 2, 3],
            "global_longest_rank": [4, 1, 3, 2],
        }
    )
    base = CompletePairCaseSelection(
        trajectory_summary=summary,
        hour_census=pd.DataFrame(
            {
                "hour_post_delivery": [2.0, 3.5, 4.5, 10.0],
                "all_bundle_count": [10, 10, 10, 10],
            }
        ),
        global_candidates=pd.DataFrame(),
        display_candidates=pd.DataFrame(),
        method_contract={},
    )
    result = exclude_acquisition_frame_counts_and_rerank(
        base, excluded_movie_frames=(5, 10), n_extreme=1
    )
    assert set(result.excluded_summary["bundle_id"]) == {"t5", "t10"}
    assert set(result.eligible_summary["bundle_id"]) == {"kept_short", "kept_long"}
    assert result.candidates.loc[0, "bundle_id"] == "kept_short"
    assert result.candidates.loc[1, "bundle_id"] == "kept_long"
