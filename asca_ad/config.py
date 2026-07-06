from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.json"
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_config() -> dict[str, dict[str, Any]]:
    return read_json(DATASET_CONFIG_PATH)


def load_experiment_config() -> dict[str, Any]:
    return read_json(EXPERIMENT_CONFIG_PATH)


def normalize_dataset_name(name: str) -> str:
    return name.strip().upper()


def select_datasets(raw: str, dataset_config: dict[str, dict[str, Any]]) -> list[str]:
    value = normalize_dataset_name(raw)

    if value == "ALL":
        return list(dataset_config.keys())

    datasets = [normalize_dataset_name(x) for x in value.split(",") if x.strip()]
    unknown = [x for x in datasets if x not in dataset_config]
    if unknown:
        valid = ", ".join(dataset_config.keys())
        raise ValueError(f"Unknown dataset(s): {unknown}. Valid datasets: {valid}")

    return datasets


def make_output_dir(raw_dataset: str, custom_output: str | None = None) -> Path:
    if custom_output:
        path = Path(custom_output)
        return path if path.is_absolute() else PROJECT_ROOT / path

    name = raw_dataset.strip().upper().replace(",", "_")
    return PROJECT_ROOT / "results" / "FIXED_COMBINED" / name
