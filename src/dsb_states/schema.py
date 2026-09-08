"""Data contracts and structured validation for DSB analysis tables.

The validators in this module are deliberately side-effect free.  They accept
in-memory tables and metadata, and return machine-readable issue records.  A
caller may explicitly write those records into an analysis run directory, but
the validators never read from or write to the production archive.

Two coordinate schemas are accepted at the structural boundary:

* canonical ``site_left/site_right`` coordinates; and
* neutral ``site1/site2`` coordinates.

Accepting the neutral schema does *not* imply a genomic flank mapping.  M0 is
closed until the mapping is explicitly documented in configuration metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The production snapshot contains a deliberately non-uniform macro-time design.
# Keep this explicit: an integer range would reject observed half-hour samples and
# silently admit hours (for example 1 h or 6 h) that are not in the current design.
DEFAULT_ALLOWED_HOURS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 9.5, 10.0, 10.5)
HIERARCHY_FIELDS = (
    "biological_experiment_id",
    "acquisition_id",
    "fov_id",
    "cell_id",
    "bundle_id",
)

FRAME_BASE_REQUIRED_FIELDS = (
    *HIERARCHY_FIELDS,
    "hour_post_delivery",
    "frame",
    "micro_time_s",
    "frame_interval_s",
    "bp1_valid",
)
CANONICAL_COORDINATE_FIELDS = (
    "site_left_x_nm",
    "site_left_y_nm",
    "site_left_valid",
    "site_right_x_nm",
    "site_right_y_nm",
    "site_right_valid",
)
NEUTRAL_COORDINATE_FIELDS = (
    "site1_x_nm",
    "site1_y_nm",
    "site1_valid",
    "site2_x_nm",
    "site2_y_nm",
    "site2_valid",
)
FRAME_REQUIRED_FIELDS = (*FRAME_BASE_REQUIRED_FIELDS, *CANONICAL_COORDINATE_FIELDS)
FRAME_NEUTRAL_REQUIRED_FIELDS = (*FRAME_BASE_REQUIRED_FIELDS, *NEUTRAL_COORDINATE_FIELDS)

BUNDLE_BASE_REQUIRED_FIELDS = (
    *HIERARCHY_FIELDS,
    "hour_post_delivery",
    "nominal_frame_count",
    "paired_valid_count",
    "paired_coverage",
    "longest_paired_run",
    "median_site_intensity",
    "bp1_detection_fraction",
    "manual_or_pipeline_qc_flags",
)
CANONICAL_BUNDLE_SITE_FIELDS = (
    "left_valid_count",
    "right_valid_count",
    "left_coverage",
    "right_coverage",
    "longest_left_run",
    "longest_right_run",
)
NEUTRAL_BUNDLE_SITE_FIELDS = (
    "site1_valid_count",
    "site2_valid_count",
    "site1_coverage",
    "site2_coverage",
    "longest_site1_run",
    "longest_site2_run",
)
BUNDLE_REQUIRED_FIELDS = (
    *HIERARCHY_FIELDS,
    "hour_post_delivery",
    "nominal_frame_count",
    "left_valid_count",
    "right_valid_count",
    "paired_valid_count",
    "left_coverage",
    "right_coverage",
    "paired_coverage",
    "longest_left_run",
    "longest_right_run",
    "longest_paired_run",
    "median_site_intensity",
    "bp1_detection_fraction",
    "manual_or_pipeline_qc_flags",
)
BUNDLE_NEUTRAL_REQUIRED_FIELDS = (
    *HIERARCHY_FIELDS,
    "hour_post_delivery",
    "nominal_frame_count",
    "site1_valid_count",
    "site2_valid_count",
    "paired_valid_count",
    "site1_coverage",
    "site2_coverage",
    "paired_coverage",
    "longest_site1_run",
    "longest_site2_run",
    "longest_paired_run",
    "median_site_intensity",
    "bp1_detection_fraction",
    "manual_or_pipeline_qc_flags",
)

# Common aliases make the contract discoverable to callers that use "columns"
# rather than "fields" terminology.
FRAME_REQUIRED_COLUMNS = FRAME_REQUIRED_FIELDS
BUNDLE_REQUIRED_COLUMNS = BUNDLE_REQUIRED_FIELDS
BUNDLE_NEUTRAL_REQUIRED_COLUMNS = BUNDLE_NEUTRAL_REQUIRED_FIELDS


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python objects."""

    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass(frozen=True)
class ValidationIssue:
    """One stable, serializable validation finding."""

    code: str
    message: str
    severity: str = "error"
    table: str | None = None
    count: int | None = None
    row_indices: tuple[Any, ...] = ()
    keys: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError(f"Unsupported issue severity: {self.severity!r}")

    @property
    def blocking(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "blocking": self.blocking,
            "table": self.table,
            "message": self.message,
            "count": self.count,
            "row_indices": _json_safe(self.row_indices),
            "keys": _json_safe(self.keys),
            "details": _json_safe(self.details),
        }


@dataclass
class ValidationReport:
    """A collection of validation issues with fail-fast helpers."""

    name: str
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def passed(self) -> bool:
        return self.valid

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)
        for key, value in other.metadata.items():
            self.metadata.setdefault(key, value)

    def has_code(self, code: str) -> bool:
        return any(issue.code == code for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        counts = {
            severity: sum(issue.severity == severity for issue in self.issues)
            for severity in ("error", "warning", "info")
        }
        return {
            "schema_version": "1.0",
            "report_name": self.name,
            "valid": self.valid,
            "issue_counts": counts,
            "metadata": _json_safe(self.metadata),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_markdown(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        lines = [
            f"# {self.name}",
            "",
            f"- Status: **{status}**",
            f"- Errors: {len(self.errors)}",
            f"- Warnings: {len(self.warnings)}",
            "",
        ]
        if not self.issues:
            return "\n".join(lines + ["No validation issues were found.", ""])
        lines.extend(["| Severity | Code | Table | Count | Message |", "|---|---|---|---:|---|"])
        for issue in self.issues:
            message = issue.message.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {issue.severity} | `{issue.code}` | {issue.table or ''} | "
                f"{'' if issue.count is None else issue.count} | {message} |"
            )
        lines.append("")
        return "\n".join(lines)

    def require_valid(self) -> None:
        if not self.valid:
            raise DataContractError(self)


class DataContractError(ValueError):
    """Raised when a caller explicitly requests fail-fast validation."""

    def __init__(self, report: ValidationReport):
        self.report = report
        codes = ", ".join(issue.code for issue in report.errors)
        super().__init__(f"{report.name} failed ({codes or 'unspecified contract error'})")


@dataclass
class GateResult:
    """Readiness and explicit activation state for a gated analysis module."""

    gate: str
    enabled: bool
    report: ValidationReport
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.report.valid

    @property
    def can_run(self) -> bool:
        return bool(self.enabled and self.ready)

    @property
    def open(self) -> bool:
        return self.can_run

    @property
    def status(self) -> str:
        if self.can_run:
            return "open"
        if not self.enabled:
            return "gated"
        return "blocked"

    @property
    def issues(self) -> list[ValidationIssue]:
        return self.report.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "enabled": self.enabled,
            "ready": self.ready,
            "can_run": self.can_run,
            "status": self.status,
            "metadata": _json_safe(self.metadata),
            "validation": self.report.to_dict(),
        }

    def require_open(self) -> None:
        if not self.can_run:
            raise GateClosedError(self)


class GateClosedError(RuntimeError):
    """Raised if a disabled or unready gated module is asked to execute."""

    def __init__(self, result: GateResult):
        self.result = result
        super().__init__(
            f"Gate {result.gate!r} is {result.status}; enabled={result.enabled}, "
            f"ready={result.ready}"
        )


def write_validation_reports(
    report: ValidationReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write the required JSON and Markdown reports to caller-selected paths."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_output.write_text(report.to_markdown(), encoding="utf-8")


def _sample_rows(table: pd.DataFrame, mask: pd.Series, columns: Sequence[str]) -> tuple:
    available = [column for column in columns if column in table.columns]
    if not available:
        return ()
    subset = table.loc[mask, available].head(20)
    return tuple(_json_safe(record) for record in subset.to_dict(orient="records"))


def _indices(table: pd.DataFrame, mask: pd.Series) -> tuple[Any, ...]:
    return tuple(_json_safe(value) for value in table.index[mask][:20].tolist())


def _missing_columns(table: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required).difference(table.columns))


def validate_required_columns(
    table: pd.DataFrame,
    required: Iterable[str],
    *,
    table_name: str,
) -> ValidationReport:
    report = ValidationReport(f"{table_name} schema validation")
    missing = _missing_columns(table, required)
    if missing:
        report.add(
            ValidationIssue(
                code="missing_required_fields",
                message=f"Required fields are absent from {table_name}: {', '.join(missing)}",
                table=table_name,
                count=len(missing),
                details={"missing_fields": missing},
            )
        )
    return report


def detect_coordinate_schema(table: pd.DataFrame) -> str | None:
    canonical = set(CANONICAL_COORDINATE_FIELDS).issubset(table.columns)
    neutral = set(NEUTRAL_COORDINATE_FIELDS).issubset(table.columns)
    if canonical:
        return "canonical_left_right"
    if neutral:
        return "neutral_site1_site2"
    return None


def detect_bundle_schema(table: pd.DataFrame) -> str | None:
    """Detect mapped or neutral single-site summaries in bundle metadata."""

    canonical = set(CANONICAL_BUNDLE_SITE_FIELDS).issubset(table.columns)
    neutral = set(NEUTRAL_BUNDLE_SITE_FIELDS).issubset(table.columns)
    if canonical:
        return "canonical_left_right"
    if neutral:
        return "neutral_site1_site2"
    return None


def _coordinate_columns(mode: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    if mode == "canonical_left_right":
        return (
            ("site_left_x_nm", "site_left_y_nm", "site_left_valid"),
            ("site_right_x_nm", "site_right_y_nm", "site_right_valid"),
        )
    if mode == "neutral_site1_site2":
        return (
            ("site1_x_nm", "site1_y_nm", "site1_valid"),
            ("site2_x_nm", "site2_y_nm", "site2_valid"),
        )
    raise ValueError(f"Unknown coordinate schema: {mode!r}")


def _bundle_site_columns(
    mode: str,
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    if mode == "canonical_left_right":
        return (
            ("left_valid_count", "left_coverage", "longest_left_run"),
            ("right_valid_count", "right_coverage", "longest_right_run"),
        )
    if mode == "neutral_site1_site2":
        return (
            ("site1_valid_count", "site1_coverage", "longest_site1_run"),
            ("site2_valid_count", "site2_coverage", "longest_site2_run"),
        )
    raise ValueError(f"Unknown bundle schema: {mode!r}")


def _coerce_boolean(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    accepted = {True, False, "true", "false", "True", "False", "0", "1"}
    invalid = series.notna() & ~series.isin(accepted)
    normalized = series.map(
        {
            True: True,
            False: False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
            "1": True,
            "0": False,
        }
    ).astype("boolean")
    return normalized, invalid


def valid_pair_mask(frame_table: pd.DataFrame) -> pd.Series:
    """Return the only valid mask for paired-locus analyses: both sites valid."""

    mode = detect_coordinate_schema(frame_table)
    if mode is None:
        raise DataContractError(
            ValidationReport(
                "valid-pair contract",
                [
                    ValidationIssue(
                        code="missing_coordinate_schema",
                        message="Neither complete left/right nor site1/site2 coordinates are present.",
                        table="frame_table",
                    )
                ],
            )
        )
    (_, _, first_valid), (_, _, second_valid) = _coordinate_columns(mode)
    first, first_invalid = _coerce_boolean(frame_table[first_valid])
    second, second_invalid = _coerce_boolean(frame_table[second_valid])
    if first.isna().any() or second.isna().any() or first_invalid.any() or second_invalid.any():
        report = ValidationReport("valid-pair contract")
        report.add(
            ValidationIssue(
                code="invalid_validity_flag",
                message="Site validity flags must be non-missing booleans before pairing.",
                table="frame_table",
            )
        )
        raise DataContractError(report)
    return (first & second).astype(bool)


def _validate_hours(
    table: pd.DataFrame,
    report: ValidationReport,
    allowed_hours: Iterable[int | float],
    table_name: str,
) -> None:
    if "hour_post_delivery" not in table:
        return
    raw = table["hour_post_delivery"]
    missing = raw.isna()
    numeric = pd.to_numeric(raw, errors="coerce")
    allowed = {float(value) for value in allowed_hours}
    if missing.any():
        report.add(
            ValidationIssue(
                code="missing_hour_post_delivery",
                message=(
                    "hour_post_delivery is missing; retain it as unknown and do not infer a "
                    "macro-time label."
                ),
                table=table_name,
                count=int(missing.sum()),
                row_indices=_indices(table, missing),
                keys=_sample_rows(
                    table,
                    missing,
                    (
                        "biological_experiment_id",
                        "acquisition_id",
                        "bundle_id",
                        "hour_post_delivery",
                    ),
                ),
                details={"field": "hour_post_delivery"},
            )
        )
    invalid = ~missing & (numeric.isna() | ~numeric.isin(allowed))
    if invalid.any():
        report.add(
            ValidationIssue(
                code="invalid_hour_post_delivery",
                message=f"hour_post_delivery must be in the configured set {sorted(allowed)}.",
                table=table_name,
                count=int(invalid.sum()),
                row_indices=_indices(table, invalid),
                keys=_sample_rows(
                    table,
                    invalid,
                    (
                        "biological_experiment_id",
                        "acquisition_id",
                        "bundle_id",
                        "hour_post_delivery",
                    ),
                ),
                details={"allowed_hours": sorted(allowed)},
            )
        )


def _validate_nonmissing_ids(
    table: pd.DataFrame,
    report: ValidationReport,
    table_name: str,
    *,
    allow_missing_biological_experiment_id: bool = False,
) -> None:
    for column in HIERARCHY_FIELDS:
        if column == "biological_experiment_id" and allow_missing_biological_experiment_id:
            continue
        if column not in table:
            continue
        values = table[column]
        missing = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        if missing.any():
            report.add(
                ValidationIssue(
                    code="missing_hierarchy_id_values",
                    message=f"Hierarchy field {column} contains missing/blank identifiers.",
                    table=table_name,
                    count=int(missing.sum()),
                    row_indices=_indices(table, missing),
                    details={"field": column},
                )
            )


def validate_frame_table(
    frame_table: pd.DataFrame,
    *,
    allowed_hours: Iterable[int | float] = DEFAULT_ALLOWED_HOURS,
    max_jump_nm: float | None = None,
    frame_interval_relative_tolerance: float = 0.05,
    allow_missing_biological_experiment_id: bool = False,
) -> ValidationReport:
    """Validate the structural and row-level frame contract.

    This function intentionally does not infer or require a site-to-flank
    mapping.  Use :func:`validate_m0_readiness` or :func:`validate_data_contract`
    for that metadata gate.
    """

    if not isinstance(frame_table, pd.DataFrame):
        raise TypeError("frame_table must be a pandas DataFrame")
    report = validate_required_columns(
        frame_table, FRAME_BASE_REQUIRED_FIELDS, table_name="frame_table"
    )
    mode = detect_coordinate_schema(frame_table)
    report.metadata.update({"row_count": len(frame_table), "coordinate_schema": mode})
    if mode is None:
        canonical_missing = _missing_columns(frame_table, CANONICAL_COORDINATE_FIELDS)
        neutral_missing = _missing_columns(frame_table, NEUTRAL_COORDINATE_FIELDS)
        report.add(
            ValidationIssue(
                code="missing_coordinate_schema",
                message=(
                    "Provide either all canonical site_left/site_right fields or all neutral "
                    "site1/site2 fields; partial coordinate schemas are not accepted."
                ),
                table="frame_table",
                details={
                    "canonical_missing_fields": canonical_missing,
                    "neutral_missing_fields": neutral_missing,
                },
            )
        )

    if report.has_code("missing_required_fields") or mode is None:
        return report

    _validate_nonmissing_ids(
        frame_table,
        report,
        "frame_table",
        allow_missing_biological_experiment_id=allow_missing_biological_experiment_id,
    )
    _validate_hours(frame_table, report, allowed_hours, "frame_table")

    duplicate = frame_table.duplicated(["bundle_id", "frame"], keep=False)
    if duplicate.any():
        report.add(
            ValidationIssue(
                code="duplicate_bundle_frame",
                message="Each (bundle_id, frame) pair must be unique.",
                table="frame_table",
                count=int(duplicate.sum()),
                row_indices=_indices(frame_table, duplicate),
                keys=_sample_rows(frame_table, duplicate, ("bundle_id", "frame")),
            )
        )

    frame_numeric = pd.to_numeric(frame_table["frame"], errors="coerce")
    invalid_frame = (
        frame_numeric.isna()
        | (frame_numeric < 0)
        | ~np.isclose(frame_numeric.fillna(0), np.round(frame_numeric.fillna(0)))
    )
    if invalid_frame.any():
        report.add(
            ValidationIssue(
                code="invalid_frame_index",
                message="frame must be a non-negative integer index.",
                table="frame_table",
                count=int(invalid_frame.sum()),
                row_indices=_indices(frame_table, invalid_frame),
            )
        )

    micro_time = pd.to_numeric(frame_table["micro_time_s"], errors="coerce")
    invalid_micro = micro_time.isna() | ~np.isfinite(micro_time)
    if invalid_micro.any():
        report.add(
            ValidationIssue(
                code="invalid_micro_time",
                message="micro_time_s must contain finite values.",
                table="frame_table",
                count=int(invalid_micro.sum()),
                row_indices=_indices(frame_table, invalid_micro),
            )
        )

    sortable = frame_table.assign(_frame=frame_numeric, _micro=micro_time)
    monotonic_failures: list[dict[str, Any]] = []
    interval_failures: list[dict[str, Any]] = []
    interval_numeric = pd.to_numeric(frame_table["frame_interval_s"], errors="coerce")
    invalid_interval = (
        interval_numeric.isna() | ~np.isfinite(interval_numeric) | (interval_numeric <= 0)
    )
    if invalid_interval.any():
        report.add(
            ValidationIssue(
                code="invalid_frame_interval",
                message="frame_interval_s must contain finite positive values.",
                table="frame_table",
                count=int(invalid_interval.sum()),
                row_indices=_indices(frame_table, invalid_interval),
            )
        )
    sortable = sortable.assign(_interval=interval_numeric)
    for bundle_id, group in sortable.groupby("bundle_id", sort=False, dropna=False):
        ordered = group.sort_values("_frame", kind="stable")
        diffs = ordered["_micro"].diff()
        bad = diffs.iloc[1:].isna() | (diffs.iloc[1:] <= 0)
        if bad.any():
            monotonic_failures.append(
                {"bundle_id": _json_safe(bundle_id), "failing_steps": int(bad.sum())}
            )
        if len(ordered) > 1 and not ordered["_interval"].isna().any():
            frame_delta = ordered["_frame"].diff().iloc[1:]
            actual_delta = ordered["_micro"].diff().iloc[1:]
            expected_delta = frame_delta * ordered["_interval"].iloc[:-1].to_numpy()
            consistent = np.isclose(
                actual_delta.to_numpy(dtype=float),
                expected_delta.to_numpy(dtype=float),
                rtol=frame_interval_relative_tolerance,
                atol=1e-6,
            )
            if (~consistent).any():
                interval_failures.append(
                    {"bundle_id": _json_safe(bundle_id), "failing_steps": int((~consistent).sum())}
                )
    if monotonic_failures:
        report.add(
            ValidationIssue(
                code="nonmonotonic_micro_time",
                message="micro_time_s must strictly increase with frame within each bundle.",
                table="frame_table",
                count=len(monotonic_failures),
                keys=tuple(monotonic_failures[:20]),
            )
        )
    if interval_failures:
        report.add(
            ValidationIssue(
                code="frame_interval_mismatch",
                message=(
                    "Observed micro-time differences disagree with frame_interval_s beyond "
                    f"the configured relative tolerance ({frame_interval_relative_tolerance:g})."
                ),
                table="frame_table",
                count=len(interval_failures),
                keys=tuple(interval_failures[:20]),
            )
        )

    for container_field in ("acquisition_id", "fov_id"):
        counts = frame_table.groupby("bundle_id", dropna=False)[container_field].nunique(
            dropna=False
        )
        crossing = counts[counts > 1]
        if not crossing.empty:
            report.add(
                ValidationIssue(
                    code=f"bundle_crosses_{container_field}",
                    message=f"A bundle must not span multiple {container_field} values.",
                    table="frame_table",
                    count=len(crossing),
                    keys=tuple(
                        {
                            "bundle_id": _json_safe(bundle_id),
                            f"n_{container_field}": int(count),
                        }
                        for bundle_id, count in crossing.head(20).items()
                    ),
                )
            )

    (x1, y1, valid1), (x2, y2, valid2) = _coordinate_columns(mode)
    normalized_valid: dict[str, pd.Series] = {}
    for x_column, y_column, valid_column in ((x1, y1, valid1), (x2, y2, valid2)):
        valid, invalid_boolean = _coerce_boolean(frame_table[valid_column])
        missing_boolean = valid.isna()
        if invalid_boolean.any() or missing_boolean.any():
            bad = invalid_boolean | missing_boolean
            report.add(
                ValidationIssue(
                    code="invalid_validity_flag",
                    message=f"{valid_column} must be a non-missing boolean/0/1 field.",
                    table="frame_table",
                    count=int(bad.sum()),
                    row_indices=_indices(frame_table, bad),
                    details={"field": valid_column},
                )
            )
        normalized_valid[valid_column] = valid.fillna(False)
        x_numeric = pd.to_numeric(frame_table[x_column], errors="coerce")
        y_numeric = pd.to_numeric(frame_table[y_column], errors="coerce")
        valid_coordinates = valid.fillna(False) & np.isfinite(x_numeric) & np.isfinite(y_numeric)
        missing_valid_coordinate = valid.fillna(False) & ~valid_coordinates
        if missing_valid_coordinate.any():
            report.add(
                ValidationIssue(
                    code="valid_site_missing_coordinates",
                    message=(
                        f"Rows marked {valid_column}=True require finite {x_column}/{y_column}."
                    ),
                    table="frame_table",
                    count=int(missing_valid_coordinate.sum()),
                    row_indices=_indices(frame_table, missing_valid_coordinate),
                    details={
                        "valid_field": valid_column,
                        "coordinate_fields": [x_column, y_column],
                    },
                )
            )
        if not valid.fillna(False).any():
            report.add(
                ValidationIssue(
                    code="all_empty_channel",
                    message=f"No valid observations are present for {valid_column}.",
                    severity="warning",
                    table="frame_table",
                    details={"valid_field": valid_column},
                )
            )

        for bundle_id, group_index in frame_table.groupby("bundle_id", dropna=False).groups.items():
            group_valid = valid.loc[group_index].fillna(False)
            if int(group_valid.sum()) < 2:
                continue
            pairs = pd.DataFrame(
                {"x": x_numeric.loc[group_index], "y": y_numeric.loc[group_index]}
            ).loc[group_valid]
            if len(pairs.drop_duplicates()) <= 1:
                report.add(
                    ValidationIssue(
                        code="constant_coordinates",
                        message=f"A bundle has constant valid coordinates for {valid_column}.",
                        severity="warning",
                        table="frame_table",
                        count=1,
                        keys=({"bundle_id": _json_safe(bundle_id), "site": valid_column},),
                    )
                )

        if max_jump_nm is not None:
            if not np.isfinite(max_jump_nm) or max_jump_nm <= 0:
                raise ValueError("max_jump_nm must be a finite positive value")
            jump_keys: list[dict[str, Any]] = []
            temp = frame_table.assign(_x=x_numeric, _y=y_numeric, _valid=valid.fillna(False))
            for bundle_id, group in temp.groupby("bundle_id", dropna=False, sort=False):
                ordered = group.sort_values("frame", kind="stable")
                adjacent = ordered["_valid"] & ordered["_valid"].shift(fill_value=False)
                jump = np.hypot(ordered["_x"].diff(), ordered["_y"].diff())
                extreme = adjacent & (jump > max_jump_nm)
                if extreme.any():
                    jump_keys.append(
                        {
                            "bundle_id": _json_safe(bundle_id),
                            "site": valid_column,
                            "extreme_jump_count": int(extreme.sum()),
                            "max_jump_nm": _json_safe(jump.loc[extreme].max()),
                        }
                    )
            if jump_keys:
                report.add(
                    ValidationIssue(
                        code="extreme_coordinate_jump",
                        message=f"Consecutive valid coordinates exceed {max_jump_nm:g} nm.",
                        severity="warning",
                        table="frame_table",
                        count=len(jump_keys),
                        keys=tuple(jump_keys[:20]),
                        details={"threshold_nm": max_jump_nm},
                    )
                )

    expected_paired = normalized_valid[valid1] & normalized_valid[valid2]
    if "paired_valid" in frame_table:
        paired, invalid_paired = _coerce_boolean(frame_table["paired_valid"])
        mismatch = invalid_paired | paired.isna() | paired.fillna(False).ne(expected_paired)
        if mismatch.any():
            report.add(
                ValidationIssue(
                    code="invalid_paired_valid_logic",
                    message="paired_valid must equal the logical AND of the two site-valid flags.",
                    table="frame_table",
                    count=int(mismatch.sum()),
                    row_indices=_indices(frame_table, mismatch),
                )
            )

    bp1_valid, invalid_bp1 = _coerce_boolean(frame_table["bp1_valid"])
    bad_bp1 = invalid_bp1 | bp1_valid.isna()
    if bad_bp1.any():
        report.add(
            ValidationIssue(
                code="invalid_bp1_valid_flag",
                message="bp1_valid must be a non-missing boolean/0/1 field.",
                table="frame_table",
                count=int(bad_bp1.sum()),
                row_indices=_indices(frame_table, bad_bp1),
            )
        )
    return report


def validate_bundle_table(
    bundle_table: pd.DataFrame,
    *,
    allowed_hours: Iterable[int | float] = DEFAULT_ALLOWED_HOURS,
    allow_missing_biological_experiment_id: bool = False,
) -> ValidationReport:
    """Validate the bundle-level metadata contract."""

    if not isinstance(bundle_table, pd.DataFrame):
        raise TypeError("bundle_table must be a pandas DataFrame")
    report = validate_required_columns(
        bundle_table, BUNDLE_BASE_REQUIRED_FIELDS, table_name="bundle_metadata"
    )
    mode = detect_bundle_schema(bundle_table)
    report.metadata.update({"row_count": len(bundle_table), "bundle_schema": mode})
    if mode is None:
        report.add(
            ValidationIssue(
                code="missing_bundle_site_schema",
                message=(
                    "Provide either all canonical left/right bundle summary fields or all "
                    "neutral site1/site2 bundle summary fields; partial schemas are not accepted."
                ),
                table="bundle_metadata",
                details={
                    "canonical_missing_fields": _missing_columns(
                        bundle_table, CANONICAL_BUNDLE_SITE_FIELDS
                    ),
                    "neutral_missing_fields": _missing_columns(
                        bundle_table, NEUTRAL_BUNDLE_SITE_FIELDS
                    ),
                },
            )
        )
    if report.has_code("missing_required_fields") or mode is None:
        return report
    _validate_nonmissing_ids(
        bundle_table,
        report,
        "bundle_metadata",
        allow_missing_biological_experiment_id=allow_missing_biological_experiment_id,
    )
    _validate_hours(bundle_table, report, allowed_hours, "bundle_metadata")

    duplicate = bundle_table.duplicated("bundle_id", keep=False)
    if duplicate.any():
        report.add(
            ValidationIssue(
                code="duplicate_bundle",
                message="bundle_metadata must contain exactly one row per bundle_id.",
                table="bundle_metadata",
                count=int(duplicate.sum()),
                row_indices=_indices(bundle_table, duplicate),
                keys=_sample_rows(bundle_table, duplicate, ("bundle_id",)),
            )
        )

    (
        (first_count, first_coverage, first_run),
        (
            second_count,
            second_coverage,
            second_run,
        ),
    ) = _bundle_site_columns(mode)
    count_fields = (
        "nominal_frame_count",
        first_count,
        second_count,
        "paired_valid_count",
        first_run,
        second_run,
        "longest_paired_run",
    )
    numeric_counts: dict[str, pd.Series] = {}
    for column in count_fields:
        numeric = pd.to_numeric(bundle_table[column], errors="coerce")
        numeric_counts[column] = numeric
        invalid = (
            numeric.isna()
            | (numeric < 0)
            | ~np.isclose(numeric.fillna(0), np.round(numeric.fillna(0)))
        )
        if column == "nominal_frame_count":
            invalid |= numeric <= 0
        if invalid.any():
            report.add(
                ValidationIssue(
                    code="invalid_bundle_count",
                    message=f"{column} must be a non-negative integer (nominal count must be >0).",
                    table="bundle_metadata",
                    count=int(invalid.sum()),
                    row_indices=_indices(bundle_table, invalid),
                    details={"field": column},
                )
            )

    nominal = numeric_counts["nominal_frame_count"]
    for column in count_fields[1:]:
        exceeds = numeric_counts[column] > nominal
        if exceeds.any():
            report.add(
                ValidationIssue(
                    code="bundle_count_exceeds_nominal",
                    message=f"{column} must not exceed nominal_frame_count.",
                    table="bundle_metadata",
                    count=int(exceeds.sum()),
                    row_indices=_indices(bundle_table, exceeds),
                    details={"field": column},
                )
            )
    if (numeric_counts["paired_valid_count"] > numeric_counts[first_count]).any() or (
        numeric_counts["paired_valid_count"] > numeric_counts[second_count]
    ).any():
        invalid = (numeric_counts["paired_valid_count"] > numeric_counts[first_count]) | (
            numeric_counts["paired_valid_count"] > numeric_counts[second_count]
        )
        report.add(
            ValidationIssue(
                code="paired_count_exceeds_site_count",
                message="paired_valid_count cannot exceed either single-site valid count.",
                table="bundle_metadata",
                count=int(invalid.sum()),
                row_indices=_indices(bundle_table, invalid),
            )
        )

    coverage_to_count = {
        first_coverage: first_count,
        second_coverage: second_count,
        "paired_coverage": "paired_valid_count",
    }
    for coverage_field, count_field in coverage_to_count.items():
        coverage = pd.to_numeric(bundle_table[coverage_field], errors="coerce")
        invalid = coverage.isna() | ~np.isfinite(coverage) | (coverage < 0) | (coverage > 1)
        if invalid.any():
            report.add(
                ValidationIssue(
                    code="invalid_bundle_coverage",
                    message=f"{coverage_field} must be a finite fraction in [0, 1].",
                    table="bundle_metadata",
                    count=int(invalid.sum()),
                    row_indices=_indices(bundle_table, invalid),
                    details={"field": coverage_field},
                )
            )
        expected = numeric_counts[count_field] / nominal
        mismatch = ~(
            np.isclose(coverage, expected, rtol=1e-6, atol=1e-8)
            | (coverage.isna() & expected.isna())
        )
        mismatch &= ~invalid & nominal.notna() & (nominal > 0)
        if mismatch.any():
            report.add(
                ValidationIssue(
                    code="bundle_coverage_count_mismatch",
                    message=f"{coverage_field} must equal {count_field}/nominal_frame_count.",
                    table="bundle_metadata",
                    count=int(mismatch.sum()),
                    row_indices=_indices(bundle_table, mismatch),
                    details={"coverage_field": coverage_field, "count_field": count_field},
                )
            )

    bp1_fraction = pd.to_numeric(bundle_table["bp1_detection_fraction"], errors="coerce")
    invalid_bp1 = bp1_fraction.notna() & (
        ~np.isfinite(bp1_fraction) | (bp1_fraction < 0) | (bp1_fraction > 1)
    )
    if invalid_bp1.any():
        report.add(
            ValidationIssue(
                code="invalid_bp1_detection_fraction",
                message="bp1_detection_fraction, when available, must lie in [0, 1].",
                table="bundle_metadata",
                count=int(invalid_bp1.sum()),
                row_indices=_indices(bundle_table, invalid_bp1),
            )
        )
    return report


def _nested(metadata: Mapping[str, Any] | None, *paths: Sequence[str]) -> Any:
    if metadata is None:
        return None
    for path in paths:
        value: Any = metadata
        found = True
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found:
            return value
    return None


def resolve_allowed_hours(
    config: Mapping[str, Any] | None = None,
    allowed_hours: Iterable[int | float] | None = None,
) -> tuple[float, ...]:
    """Resolve an explicit override, configured macro hours, or snapshot default.

    Configuration may use ``time_axes.allowed_hours`` (preferred) or one of the
    supported descriptive aliases below.  Invalid or empty declarations fail
    fast rather than causing hours to be inferred from the observed table.
    """

    selected: Any = allowed_hours
    if selected is None:
        selected = _nested(
            config,
            ("time_axes", "allowed_hours"),
            ("time_axes", "allowed_macro_hours"),
            ("time_axes", "allowed_hours_post_delivery"),
            ("macro_time", "allowed_hours"),
            ("validation", "allowed_hours"),
            ("allowed_hours",),
        )
    if selected is None:
        selected = DEFAULT_ALLOWED_HOURS
    if isinstance(selected, (str, bytes, Mapping)):
        raise TypeError("allowed_hours must be a non-empty iterable of finite numeric hours")
    try:
        resolved = tuple(float(value) for value in selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "allowed_hours must be a non-empty iterable of finite numeric hours"
        ) from exc
    if not resolved or not all(np.isfinite(value) for value in resolved):
        raise ValueError("allowed_hours must be a non-empty iterable of finite numeric hours")
    return tuple(sorted(set(resolved)))


def _is_resolved_text(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in {"", "none", "null", "unknown", "unresolved", "tbd", "na", "n/a"}


def validate_m0_readiness(
    frame_table: pd.DataFrame,
    bundle_table: pd.DataFrame | None,
    config: Mapping[str, Any] | None,
) -> ValidationReport:
    """Validate M0 metadata without inferring missing biological identities."""

    report = ValidationReport("M0 data-readiness validation")
    mode = detect_coordinate_schema(frame_table)
    report.metadata["coordinate_schema"] = mode
    channels = config.get("channels", {}) if isinstance(config, Mapping) else {}
    mapping_status = channels.get("flank_mapping_status") if isinstance(channels, Mapping) else None
    left_source = channels.get("site_left_source") if isinstance(channels, Mapping) else None
    right_source = channels.get("site_right_source") if isinstance(channels, Mapping) else None
    valid_sources = {"site1", "site2"}
    mapping_is_explicit = (
        str(mapping_status).strip().lower() in {"resolved", "verified", "confirmed"}
        and str(left_source).strip().lower() in valid_sources
        and str(right_source).strip().lower() in valid_sources
        and str(left_source).strip().lower() != str(right_source).strip().lower()
    )
    if not mapping_is_explicit:
        report.add(
            ValidationIssue(
                code="m0_flank_mapping_unresolved",
                message=(
                    "M0 is closed: explicitly resolve Site1/Site2 to cut-left/cut-right in "
                    "channels.flank_mapping_status/site_left_source/site_right_source."
                ),
                table="metadata",
                details={
                    "flank_mapping_status": _json_safe(mapping_status),
                    "site_left_source": _json_safe(left_source),
                    "site_right_source": _json_safe(right_source),
                },
            )
        )

    unit = _nested(
        config,
        ("units", "coordinate_unit"),
        ("units", "output_coordinate_unit"),
        ("coordinates", "units"),
        ("coordinates", "unit"),
        ("data_dictionary", "coordinate_unit"),
    )
    nm_per_px = _nested(
        config,
        ("units", "nm_per_px"),
        ("coordinates", "nm_per_px"),
        ("data_dictionary", "nm_per_px"),
    )
    direct_nm = isinstance(unit, str) and unit.strip().lower() in {"nm", "nanometer", "nanometers"}
    conversion = False
    try:
        conversion = bool(float(nm_per_px) > 0 and np.isfinite(float(nm_per_px)))
    except (TypeError, ValueError):
        conversion = False
    if not (direct_nm or conversion):
        report.add(
            ValidationIssue(
                code="m0_nm_unit_evidence_missing",
                message=(
                    "M0 is closed: document nm output units or a finite positive nm_per_px "
                    "conversion in configuration/data-dictionary metadata."
                ),
                table="metadata",
                details={"coordinate_unit": _json_safe(unit), "nm_per_px": _json_safe(nm_per_px)},
            )
        )

    coordinate_alignment = _nested(
        config,
        ("coordinates", "bp1_dna_same_coordinate_system"),
        ("channels", "bp1_dna_same_coordinate_system"),
        ("data_dictionary", "bp1_dna_same_coordinate_system"),
    )
    if coordinate_alignment is not True:
        report.add(
            ValidationIssue(
                code="m0_bp1_coordinate_system_unverified",
                message="M0 is closed until BP1 and DNA coordinates are documented as co-registered.",
                table="metadata",
                details={"bp1_dna_same_coordinate_system": _json_safe(coordinate_alignment)},
            )
        )

    hierarchy = config.get("hierarchy", {}) if isinstance(config, Mapping) else {}
    biological_unavailable = bool(
        isinstance(hierarchy, Mapping)
        and hierarchy.get("biological_experiment_id_source") == "unavailable_by_design"
    )
    if biological_unavailable:
        declared_design = bool(
            hierarchy.get("analysis_group_id_source") == "acquisition_id"
            and hierarchy.get("independent_biological_replicates_available") is False
            and str(hierarchy.get("biological_population_inference", "")).strip().lower()
            == "prohibited"
        )
        if not declared_design:
            report.add(
                ValidationIssue(
                    code="m0_unavailable_biological_group_contract_invalid",
                    message=(
                        "unavailable_by_design requires acquisition_id grouping, an explicit "
                        "false independent-replicate flag, and prohibited population inference."
                    ),
                    table="metadata",
                    details={"field": "biological_experiment_id"},
                )
            )
    for hierarchy_field in HIERARCHY_FIELDS[:-1]:
        source_key = f"{hierarchy_field}_source"
        # bundle_id is often a composed key and is checked from the table itself.
        if hierarchy_field == "biological_experiment_id":
            value = hierarchy.get(source_key) if isinstance(hierarchy, Mapping) else None
            if not _is_resolved_text(value):
                report.add(
                    ValidationIssue(
                        code="m0_hierarchy_source_unresolved",
                        message=f"M0 is closed: {source_key} is not documented.",
                        table="metadata",
                        details={"field": hierarchy_field, "source": _json_safe(value)},
                    )
                )
    if bundle_table is not None and set(HIERARCHY_FIELDS).issubset(bundle_table.columns):
        for field in ("acquisition_id", "fov_id"):
            if bundle_table[field].isna().any():
                report.add(
                    ValidationIssue(
                        code="m0_required_grouping_id_missing",
                        message=f"M0 is closed because {field} is missing for one or more bundles.",
                        table="bundle_metadata",
                        count=int(bundle_table[field].isna().sum()),
                        details={"field": field},
                    )
                )
    return report


def validate_data_contract(
    frame_table: pd.DataFrame,
    bundle_table: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None,
    allowed_hours: Iterable[int | float] | None = None,
    max_jump_nm: float | None = None,
) -> ValidationReport:
    """Validate frame/bundle schemas, reconciliation, and the full M0 gate."""

    resolved_allowed_hours = resolve_allowed_hours(config, allowed_hours)
    report = ValidationReport("DSB data-contract validation")
    hierarchy = config.get("hierarchy", {}) if isinstance(config, Mapping) else {}
    biological_unavailable = bool(
        isinstance(hierarchy, Mapping)
        and hierarchy.get("biological_experiment_id_source") == "unavailable_by_design"
        and hierarchy.get("analysis_group_id_source") == "acquisition_id"
        and hierarchy.get("independent_biological_replicates_available") is False
        and str(hierarchy.get("biological_population_inference", "")).strip().lower()
        == "prohibited"
    )
    frame_report = validate_frame_table(
        frame_table,
        allowed_hours=resolved_allowed_hours,
        max_jump_nm=max_jump_nm,
        allow_missing_biological_experiment_id=biological_unavailable,
    )
    bundle_report = validate_bundle_table(
        bundle_table,
        allowed_hours=resolved_allowed_hours,
        allow_missing_biological_experiment_id=biological_unavailable,
    )
    report.extend(frame_report)
    report.extend(bundle_report)
    report.metadata.update(
        {
            "frame_table_rows": len(frame_table),
            "bundle_table_rows": len(bundle_table),
            "coordinate_schema": detect_coordinate_schema(frame_table),
            "bundle_schema": detect_bundle_schema(bundle_table),
            "allowed_hours": list(resolved_allowed_hours),
        }
    )

    if "bundle_id" in frame_table and "bundle_id" in bundle_table:
        frame_ids = set(frame_table["bundle_id"].dropna().tolist())
        bundle_ids = set(bundle_table["bundle_id"].dropna().tolist())
        absent_metadata = frame_ids - bundle_ids
        absent_frames = bundle_ids - frame_ids
        if absent_metadata:
            report.add(
                ValidationIssue(
                    code="frame_bundle_missing_metadata",
                    message="Frame-table bundles are absent from bundle_metadata.",
                    table="cross_table",
                    count=len(absent_metadata),
                    keys=tuple(
                        {"bundle_id": _json_safe(value)} for value in list(absent_metadata)[:20]
                    ),
                )
            )
        if absent_frames:
            report.add(
                ValidationIssue(
                    code="metadata_bundle_missing_frames",
                    message="Bundle-metadata rows have no corresponding frame rows.",
                    table="cross_table",
                    count=len(absent_frames),
                    keys=tuple(
                        {"bundle_id": _json_safe(value)} for value in list(absent_frames)[:20]
                    ),
                )
            )

    mode = detect_coordinate_schema(frame_table)
    bundle_mode = detect_bundle_schema(bundle_table)
    required_cross = set(HIERARCHY_FIELDS).issubset(frame_table.columns) and set(
        HIERARCHY_FIELDS
    ).issubset(bundle_table.columns)
    if (
        mode is not None
        and bundle_mode is not None
        and required_cross
        and not bundle_table.duplicated("bundle_id").any()
    ):
        (_, _, valid1), (_, _, valid2) = _coordinate_columns(mode)
        first, first_invalid = _coerce_boolean(frame_table[valid1])
        second, second_invalid = _coerce_boolean(frame_table[valid2])
        if not (
            first_invalid.any() or second_invalid.any() or first.isna().any() or second.isna().any()
        ):
            summary = (
                frame_table.assign(
                    _first=first.astype(bool),
                    _second=second.astype(bool),
                    _paired=(first & second).astype(bool),
                )
                .groupby("bundle_id", dropna=False)
                .agg(
                    _first_count=("_first", "sum"),
                    _second_count=("_second", "sum"),
                    _paired_count=("_paired", "sum"),
                )
            )
            metadata = bundle_table.set_index("bundle_id")
            common = summary.index.intersection(metadata.index)
            count_sources = {"paired_valid_count": "_paired_count"}
            (bundle_first_count, _, _), (bundle_second_count, _, _) = _bundle_site_columns(
                bundle_mode
            )
            if mode == bundle_mode:
                count_sources.update(
                    {
                        bundle_first_count: "_first_count",
                        bundle_second_count: "_second_count",
                    }
                )
            else:
                channels = config.get("channels", {}) if isinstance(config, Mapping) else {}
                left_source = str(channels.get("site_left_source", "")).strip().lower()
                right_source = str(channels.get("site_right_source", "")).strip().lower()
                mapping_resolved = {left_source, right_source} == {"site1", "site2"}
                if mapping_resolved and mode == "neutral_site1_site2":
                    neutral_sources = {"site1": "_first_count", "site2": "_second_count"}
                    count_sources.update(
                        {
                            "left_valid_count": neutral_sources[left_source],
                            "right_valid_count": neutral_sources[right_source],
                        }
                    )
                elif mapping_resolved and mode == "canonical_left_right":
                    canonical_sources = {"left": "_first_count", "right": "_second_count"}
                    site1_side = "left" if left_source == "site1" else "right"
                    site2_side = "left" if left_source == "site2" else "right"
                    count_sources.update(
                        {
                            "site1_valid_count": canonical_sources[site1_side],
                            "site2_valid_count": canonical_sources[site2_side],
                        }
                    )
                else:
                    report.add(
                        ValidationIssue(
                            code="cross_table_single_site_reconciliation_skipped",
                            message=(
                                "Frame and bundle site schemas differ, and no explicit flank "
                                "mapping is available; only paired counts were reconciled."
                            ),
                            severity="warning",
                            table="cross_table",
                            details={
                                "frame_schema": mode,
                                "bundle_schema": bundle_mode,
                            },
                        )
                    )
            for column, summary_column in count_sources.items():
                if column not in metadata:
                    continue
                observed = pd.to_numeric(metadata.loc[common, column], errors="coerce")
                expected = summary.loc[common, summary_column]
                mismatch = observed.ne(expected)
                if mismatch.any():
                    keys = tuple(
                        {
                            "bundle_id": _json_safe(bundle_id),
                            "metadata_value": _json_safe(observed.loc[bundle_id]),
                            "frame_derived_value": _json_safe(expected.loc[bundle_id]),
                        }
                        for bundle_id in common[mismatch][:20]
                    )
                    report.add(
                        ValidationIssue(
                            code="frame_bundle_count_mismatch",
                            message=f"Frame-derived {column} disagrees with bundle_metadata.",
                            table="cross_table",
                            count=int(mismatch.sum()),
                            keys=keys,
                            details={"field": column},
                        )
                    )

        common_fields = (
            "biological_experiment_id",
            "acquisition_id",
            "fov_id",
            "cell_id",
            "hour_post_delivery",
        )
        frame_first = frame_table.groupby("bundle_id", dropna=False)[list(common_fields)].first()
        metadata = bundle_table.set_index("bundle_id")
        common = frame_first.index.intersection(metadata.index)
        for column in common_fields:
            mismatch = (
                frame_first.loc[common, column]
                .astype("string")
                .ne(metadata.loc[common, column].astype("string"))
            )
            if mismatch.any():
                report.add(
                    ValidationIssue(
                        code="frame_bundle_hierarchy_mismatch",
                        message=f"Frame and bundle tables disagree on {column}.",
                        table="cross_table",
                        count=int(mismatch.sum()),
                        keys=tuple(
                            {"bundle_id": _json_safe(bundle_id), "field": column}
                            for bundle_id in common[mismatch][:20]
                        ),
                    )
                )

    report.extend(validate_m0_readiness(frame_table, bundle_table, config))
    return report


def partition_validation_errors(
    report: ValidationReport, m0_blocking_contracts: Iterable[str]
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Split expected metadata-gate findings from structural table failures.

    Only the three missing metadata conditions that leave neutral T03 QC
    scientifically interpretable are eligible for the first partition. Unit,
    timing, coordinate, pairing, or acquisition/FOV errors always remain
    structural and must stop QC.
    """

    blockers = set(m0_blocking_contracts)
    expected_metadata: list[ValidationIssue] = []
    structural: list[ValidationIssue] = []
    for issue in report.errors:
        field = str(issue.details.get("field", ""))
        contract: str | None = None
        if issue.code == "missing_hour_post_delivery":
            contract = "macro_time_mapping"
        elif (
            issue.code == "missing_hierarchy_id_values"
            and field == "biological_experiment_id"
            or issue.code == "m0_hierarchy_source_unresolved"
            and field == "biological_experiment_id"
        ):
            contract = "biological_experiment_id"
        elif issue.code == "m0_flank_mapping_unresolved":
            contract = "site1_site2_flank_mapping"
        destination = (
            expected_metadata if contract is not None and contract in blockers else structural
        )
        destination.append(issue)
    return expected_metadata, structural


# Explicit aliases for callers using "metadata" or "all tables" terminology.
validate_bundle_metadata = validate_bundle_table
validate_analysis_tables = validate_data_contract
