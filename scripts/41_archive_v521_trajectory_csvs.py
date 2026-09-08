#!/usr/bin/env python3
"""Build a CSV-only archive of final v5.2.1 Site1/Site2/53BP1 trajectories.

The production full run is strictly read-only.  This script uses the frozen crop
audit to find crops with a formal ``baseline_longest`` manifest, copies every
declared cleaned Site1/Site2/53BP1 trajectory, and archives them as::

    by_nd2/<nd2_id>/<crop_id>/<crop_id>__allele_###__<marker>__trajectory.csv

No scientific QC, pair requirement, hour filter, motion filter, separation
filter, or 53BP1-presence filter is applied.  A 53BP1 trajectory is archived
when the formal baseline manifest declares one; its absence is allowed.  Raw
SPT candidates and diagnostic CSVs are not part of this final-trajectory archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

TRACK_NAME = re.compile(
    r"^allele_(\d+)_(site1|site2|53bp1)_longest_spt_cleaned\.csv$"
)
INVALID_WINDOWS = re.compile(r'[<>:"/\\|?*]+')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def safe_component(value: str) -> str:
    value = INVALID_WINDOWS.sub("_", value.strip())
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.rstrip(". ")


def localize_fullrun_path(path_text: str, source_root: Path) -> Path:
    normalized = path_text.replace("\\", "/")
    marker = f"/{source_root.name}/"
    if marker not in normalized:
        raise ValueError(f"Cannot localize path outside frozen full run: {path_text}")
    relative = normalized.split(marker, maxsplit=1)[1]
    return source_root.joinpath(*PurePosixPath(relative).parts)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_trajectory_with_sha256(
    source: Path, destination: Path
) -> tuple[str, int, str]:
    """Copy and validate a trajectory in one source-file read."""

    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        header_bytes = incoming.readline()
        if not header_bytes:
            raise ValueError(f"Empty trajectory CSV: {source}")
        digest.update(header_bytes)
        outgoing.write(header_bytes)
        rows = 0
        for line in incoming:
            digest.update(line)
            outgoing.write(line)
            if line.strip():
                rows += 1
    shutil.copystat(source, destination)
    if source.stat().st_size != destination.stat().st_size:
        raise OSError(f"Copied size differs from source: {source}")
    columns = "|".join(
        part.strip()
        for part in header_bytes.decode("utf-8-sig").strip("\r\n").split(",")
    )
    return digest.hexdigest(), rows, columns


def write_dict_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def archive_crop(
    crop: dict[str, str], source_root: Path, staging: Path
) -> list[dict[str, object]]:
    result_dir = localize_fullrun_path(crop["result_dir"], source_root)
    baseline_dir = result_dir / "baseline_longest"
    baseline_manifest = baseline_dir / "baseline_manifest.csv"
    if not baseline_manifest.is_file():
        raise FileNotFoundError(baseline_manifest)

    output: list[dict[str, object]] = []
    relative_keys: set[str] = set()
    for declared in read_csv(baseline_manifest):
        source_name = Path(declared.get("baseline_csv", "")).name
        match = TRACK_NAME.fullmatch(source_name)
        if match is None:
            raise ValueError(
                f"Unexpected formal baseline trajectory name {source_name!r} in "
                f"{baseline_manifest}"
            )
        allele_index = int(match.group(1))
        marker_slug = match.group(2)
        if int(declared["allele_index"]) != allele_index:
            raise ValueError(f"Allele mismatch in {baseline_manifest}: {source_name}")
        if declared["site_id"] != marker_slug:
            raise ValueError(f"Marker mismatch in {baseline_manifest}: {source_name}")

        source_csv = baseline_dir / source_name
        if not source_csv.is_file():
            raise FileNotFoundError(source_csv)
        nd2_slug = safe_component(crop["nd2_id"])
        crop_slug = safe_component(crop["crop_id"])
        archive_name = (
            f"{crop_slug}__allele_{allele_index:03d}__{marker_slug}__trajectory.csv"
        )
        relative_archive = Path("by_nd2") / nd2_slug / crop_slug / archive_name
        relative_key = relative_archive.as_posix().casefold()
        if relative_key in relative_keys:
            raise ValueError(f"Archive naming collision: {relative_archive}")
        relative_keys.add(relative_key)

        destination = staging / relative_archive
        sha256, n_rows, columns = copy_trajectory_with_sha256(source_csv, destination)
        expected_points = int(float(declared["points"]))
        if n_rows != expected_points:
            raise ValueError(
                f"Row count {n_rows} != manifest points {expected_points}: {source_csv}"
            )
        if columns != "frame|x_nm|y_nm":
            raise ValueError(f"Unexpected trajectory columns {columns!r}: {source_csv}")
        output.append(
            {
                "archive_file": relative_archive.as_posix(),
                "cohort": crop.get("cohort", ""),
                "nd2_id": crop.get("nd2_id", ""),
                "crop_id": crop.get("crop_id", ""),
                "crop_index": crop.get("crop_index", ""),
                "task_status": crop.get("task_status", ""),
                "allele_index": allele_index,
                "site_id": marker_slug,
                "source_candidate_number": declared.get("candidate_number", ""),
                "trajectory_points": n_rows,
                "first_frame": declared.get("first_frame", ""),
                "last_frame": declared.get("last_frame", ""),
                "frame_span": declared.get("frame_span", ""),
                "temporal_coverage_fraction": declared.get(
                    "temporal_coverage_fraction", ""
                ),
                "maximum_missing_frames_between_points": declared.get(
                    "maximum_missing_frames_between_points", ""
                ),
                "frame_interval_s": declared.get("frame_interval_s", ""),
                "movie_frame_count": declared.get("movie_frame_count", crop.get("frames", "")),
                "pixel_size_nm_per_px": declared.get("pixel_size_nm_per_px", ""),
                "columns": columns,
                "bytes": source_csv.stat().st_size,
                "sha256": sha256,
                "source_file": str(source_csv),
                "source_baseline_manifest": str(baseline_manifest),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    audit_path = source_root / "06_reports" / "crop_final_audit.tsv"

    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output_root}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    staging = output_root.with_name(f"{output_root.name}.building_{timestamp}_{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    audit_rows = read_tsv(audit_path)
    archive_rows: list[dict[str, object]] = []
    unavailable_rows: list[dict[str, object]] = []
    available_crops: list[dict[str, str]] = []

    for crop in audit_rows:
        if crop.get("baseline_manifest_present") != "1":
            unavailable_rows.append(
                {
                    "cohort": crop.get("cohort", ""),
                    "nd2_id": crop.get("nd2_id", ""),
                    "crop_id": crop.get("crop_id", ""),
                    "crop_index": crop.get("crop_index", ""),
                    "movie_frames": crop.get("frames", ""),
                    "task_status": crop.get("task_status", ""),
                    "reason": "formal_baseline_manifest_absent",
                    "source_result_dir": crop.get("result_dir", ""),
                }
            )
            continue
        available_crops.append(crop)

    crop_keys = [(crop["nd2_id"], crop["crop_id"]) for crop in available_crops]
    if len(crop_keys) != len(set(crop_keys)):
        raise ValueError("Duplicate ND2/crop identities in frozen crop audit")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    copied = 0
    next_progress = args.progress_every
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(archive_crop, crop, source_root, staging): crop
            for crop in available_crops
        }
        for future in as_completed(futures):
            crop = futures[future]
            rows = future.result()
            if not rows:
                unavailable_rows.append(
                    {
                        "cohort": crop.get("cohort", ""),
                        "nd2_id": crop.get("nd2_id", ""),
                        "crop_id": crop.get("crop_id", ""),
                        "crop_index": crop.get("crop_index", ""),
                        "movie_frames": crop.get("frames", ""),
                        "task_status": crop.get("task_status", ""),
                        "reason": "formal_baseline_manifest_empty",
                        "source_result_dir": crop.get("result_dir", ""),
                    }
                )
            archive_rows.extend(rows)
            copied += len(rows)
            if copied >= next_progress:
                print(f"copied={copied}", flush=True)
                while next_progress <= copied:
                    next_progress += args.progress_every

    archive_rows.sort(
        key=lambda row: (
            str(row["nd2_id"]),
            str(row["crop_id"]),
            int(row["allele_index"]),
            str(row["site_id"]),
        )
    )
    unavailable_rows.sort(key=lambda row: (str(row["nd2_id"]), str(row["crop_id"])))

    manifest_fields = [
        "archive_file",
        "cohort",
        "nd2_id",
        "crop_id",
        "crop_index",
        "task_status",
        "allele_index",
        "site_id",
        "source_candidate_number",
        "trajectory_points",
        "first_frame",
        "last_frame",
        "frame_span",
        "temporal_coverage_fraction",
        "maximum_missing_frames_between_points",
        "frame_interval_s",
        "movie_frame_count",
        "pixel_size_nm_per_px",
        "columns",
        "bytes",
        "sha256",
        "source_file",
        "source_baseline_manifest",
    ]
    write_dict_csv(staging / "trajectory_manifest.csv", archive_rows, manifest_fields)
    write_dict_csv(
        staging / "unavailable_crops.csv",
        unavailable_rows,
        [
            "cohort",
            "nd2_id",
            "crop_id",
            "crop_index",
            "movie_frames",
            "task_status",
            "reason",
            "source_result_dir",
        ],
    )

    sites = Counter(str(row["site_id"]) for row in archive_rows)
    statuses = Counter(str(row["task_status"]) for row in archive_rows)
    unavailable_reasons = Counter(str(row["reason"]) for row in unavailable_rows)
    crops_with_tracks = len({(row["nd2_id"], row["crop_id"]) for row in archive_rows})
    summary_rows = [
        {"key": "archive_schema", "value": "v521_final_cleaned_trajectory_csv_v1"},
        {"key": "created_utc", "value": timestamp},
        {"key": "source_root", "value": str(source_root)},
        {"key": "source_version", "value": "5.2.1"},
        {"key": "source_crop_audit", "value": str(audit_path)},
        {"key": "source_crops_total", "value": len(audit_rows)},
        {
            "key": "source_crops_with_formal_baseline",
            "value": len(available_crops),
        },
        {"key": "source_crops_with_trajectory_csv", "value": crops_with_tracks},
        {
            "key": "source_crops_without_formal_baseline",
            "value": unavailable_reasons.get("formal_baseline_manifest_absent", 0),
        },
        {
            "key": "source_crops_with_empty_formal_baseline",
            "value": unavailable_reasons.get("formal_baseline_manifest_empty", 0),
        },
        {"key": "source_crops_without_any_trajectory", "value": len(unavailable_rows)},
        {"key": "trajectory_csv_total", "value": len(archive_rows)},
        {"key": "site1_trajectory_csv", "value": sites.get("site1", 0)},
        {"key": "site2_trajectory_csv", "value": sites.get("site2", 0)},
        {"key": "53bp1_trajectory_csv", "value": sites.get("53bp1", 0)},
        {
            "key": "trajectory_rows_total",
            "value": sum(int(row["trajectory_points"]) for row in archive_rows),
        },
        {"key": "scientific_qc_applied", "value": "none"},
        {"key": "pair_requirement_applied", "value": "false"},
        {"key": "53bp1_filter_applied", "value": "false"},
        {
            "key": "included_roles",
            "value": "formal baseline_longest Site1/Site2/53BP1 cleaned trajectories",
        },
        {
            "key": "excluded_roles",
            "value": "raw SPT candidates; diagnostic CSVs; 53BP1 metric tables",
        },
    ]
    summary_rows.extend(
        {"key": f"trajectory_file_task_status_{key}", "value": value}
        for key, value in sorted(statuses.items())
    )
    write_dict_csv(staging / "archive_summary.csv", summary_rows, ["key", "value"])

    actual_csvs = sum(1 for path in staging.rglob("*.csv") if path.is_file())
    expected_csvs = len(archive_rows) + 3
    if actual_csvs != expected_csvs:
        raise ValueError(f"Archive CSV count mismatch: {actual_csvs} != {expected_csvs}")

    staging.rename(output_root)
    print(f"archive={output_root}")
    print(f"trajectory_csvs={len(archive_rows)}")
    print(f"trajectory_rows={sum(int(row['trajectory_points']) for row in archive_rows)}")
    print(f"unavailable_crops={len(unavailable_rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
