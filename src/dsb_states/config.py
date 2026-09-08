from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the analysis configuration is inconsistent or incomplete."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(raw)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent.resolve())

    data = config.setdefault("data", {})
    env_name = data.get("archive_root_env", "DSB_V51_ROOT")
    archive_root = os.environ.get(env_name) or data.get("archive_root_default")
    if not archive_root:
        raise ConfigurationError(
            f"Set {env_name} or data.archive_root_default; source paths are never guessed."
        )
    data["archive_root_resolved"] = str(Path(archive_root).expanduser().resolve())
    return config


def write_resolved_config(config: dict[str, Any], output_path: str | Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
