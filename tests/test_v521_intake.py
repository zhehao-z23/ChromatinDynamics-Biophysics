from pathlib import Path

from dsb_states.v521_intake import _localize, classify_availability


def test_localize_sherlock_path_to_frozen_root(tmp_path: Path) -> None:
    root = tmp_path / "dsb_v521_fullrun_20260819T084304Z"
    source = (
        "/scratch/work/dsb_v521_fullrun_20260819T084304Z/03_new_all_candidates/acquisition/crop.csv"
    )
    assert _localize(source, root) == root / "03_new_all_candidates/acquisition/crop.csv"


def test_pair_with_shared_frames_is_separate_from_track_only_pair() -> None:
    assert (
        classify_availability(
            site1=True,
            site2=True,
            paired_tracks=True,
            paired_frames=True,
            bp1_global=True,
            bp1_locus=True,
            bp1_spt=False,
        )
        == "PAIRED_FRAMES_PLUS_LOCUS_BP1"
    )
    assert (
        classify_availability(
            site1=True,
            site2=True,
            paired_tracks=True,
            paired_frames=False,
            bp1_global=True,
            bp1_locus=True,
            bp1_spt=False,
        )
        == "PAIRED_TRACKS_NO_SHARED_FRAMES_PLUS_LOCUS_BP1"
    )


def test_global_bp1_only_is_not_called_unusable() -> None:
    assert (
        classify_availability(
            site1=False,
            site2=False,
            paired_tracks=False,
            paired_frames=False,
            bp1_global=True,
            bp1_locus=False,
            bp1_spt=False,
        )
        == "BP1_ONLY"
    )
