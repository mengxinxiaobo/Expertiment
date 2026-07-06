from __future__ import annotations

from asca_ad.config import load_dataset_config


def available_datasets() -> list[str]:
    return list(load_dataset_config().keys())


def get_dataset_config(name: str) -> dict:
    config = load_dataset_config()
    key = name.strip().upper()
    if key not in config:
        valid = ", ".join(config.keys())
        raise ValueError(f"Unknown dataset: {key}. Valid datasets: {valid}")
    return config[key]
