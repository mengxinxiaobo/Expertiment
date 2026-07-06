from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_dataset_config


@dataclass(frozen=True)
class DatasetRuntime:
    name: str
    epochs: int
    anormly_ratio: float
    runner: str

    @property
    def runner_path(self) -> Path:
        return PROJECT_ROOT / "scripts" / "dataset_runners" / self.runner


def get_dataset_runtime(name: str) -> DatasetRuntime:
    config = load_dataset_config()
    key = name.strip().upper()
    if key not in config:
        valid = ", ".join(config.keys())
        raise ValueError(f"Unknown dataset: {key}. Valid datasets: {valid}")

    item: dict[str, Any] = config[key]
    return DatasetRuntime(
        name=key,
        epochs=int(item.get("epochs", 3)),
        anormly_ratio=float(item["anormly_ratio"]),
        runner=str(item["runner"]),
    )


def list_datasets() -> list[str]:
    return list(load_dataset_config().keys())
