#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


FILES = {
    "asca_ad/config.py": 'from __future__ import annotations\n\nimport json\nfrom pathlib import Path\nfrom typing import Any\n\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nDATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.json"\nEXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"\n\n\ndef read_json(path: str | Path) -> dict[str, Any]:\n    path = Path(path)\n    if not path.exists():\n        raise FileNotFoundError(f"Missing config file: {path}")\n    return json.loads(path.read_text(encoding="utf-8"))\n\n\ndef load_dataset_config() -> dict[str, dict[str, Any]]:\n    return read_json(DATASET_CONFIG_PATH)\n\n\ndef load_experiment_config() -> dict[str, Any]:\n    return read_json(EXPERIMENT_CONFIG_PATH)\n\n\ndef normalize_dataset_name(name: str) -> str:\n    return name.strip().upper()\n\n\ndef select_datasets(raw: str, dataset_config: dict[str, dict[str, Any]]) -> list[str]:\n    value = normalize_dataset_name(raw)\n\n    if value == "ALL":\n        return list(dataset_config.keys())\n\n    datasets = [normalize_dataset_name(x) for x in value.split(",") if x.strip()]\n    unknown = [x for x in datasets if x not in dataset_config]\n    if unknown:\n        valid = ", ".join(dataset_config.keys())\n        raise ValueError(f"Unknown dataset(s): {unknown}. Valid datasets: {valid}")\n\n    return datasets\n\n\ndef make_output_dir(raw_dataset: str, custom_output: str | None = None) -> Path:\n    if custom_output:\n        path = Path(custom_output)\n        return path if path.is_absolute() else PROJECT_ROOT / path\n\n    name = raw_dataset.strip().upper().replace(",", "_")\n    return PROJECT_ROOT / "results" / "FIXED_COMBINED" / name\n',
    "asca_ad/evaluator.py": 'from __future__ import annotations\n\nimport importlib.util\nimport json\nimport sys\nfrom pathlib import Path\nfrom typing import Any\n\nfrom .config import (\n    PROJECT_ROOT,\n    load_dataset_config,\n    load_experiment_config,\n    make_output_dir,\n    select_datasets,\n)\n\n\nENGINE_PATH = PROJECT_ROOT / "scripts" / "run_all_fixed_combined.py"\n\n\ndef load_fixed_combined_engine():\n    """Load the existing fixed-combined evaluation engine.\n\n    This keeps backward compatibility with the current validated implementation\n    while moving the public evaluation API into the asca_ad package.\n    """\n    if not ENGINE_PATH.exists():\n        raise FileNotFoundError(f"Missing evaluation engine: {ENGINE_PATH}")\n\n    spec = importlib.util.spec_from_file_location(\n        "asca_ad_fixed_combined_engine",\n        str(ENGINE_PATH),\n    )\n    if spec is None or spec.loader is None:\n        raise RuntimeError(f"Cannot import evaluation engine: {ENGINE_PATH}")\n\n    module = importlib.util.module_from_spec(spec)\n    sys.modules["asca_ad_fixed_combined_engine"] = module\n    spec.loader.exec_module(module)\n    return module\n\n\ndef run_evaluation(\n    dataset: str,\n    output_dir: str | None = None,\n    seed: int | None = None,\n    epochs: int | None = None,\n    train_if_missing: bool = False,\n) -> list[dict[str, Any]]:\n    dataset_config = load_dataset_config()\n    experiment_config = load_experiment_config()\n    engine = load_fixed_combined_engine()\n\n    datasets = select_datasets(dataset, dataset_config)\n    out_dir = make_output_dir(dataset, output_dir)\n    out_dir.mkdir(parents=True, exist_ok=True)\n\n    final_seed = (\n        int(seed)\n        if seed is not None\n        else int(experiment_config.get("default_seed", 42))\n    )\n\n    print("=" * 80)\n    print("ASCA-AD fixed-combined evaluation")\n    print("=" * 80)\n    print(f"Project root : {PROJECT_ROOT}")\n    print(f"Datasets     : {\', \'.join(datasets)}")\n    print(f"Output dir   : {out_dir}")\n    print(f"Seed         : {final_seed}")\n    print("=" * 80)\n\n    results: list[dict[str, Any]] = []\n    failures: list[dict[str, str]] = []\n\n    for item in datasets:\n        cfg = dataset_config[item]\n        ratio = float(cfg["anormly_ratio"])\n        final_epochs = int(epochs) if epochs is not None else int(cfg.get("epochs", 3))\n\n        try:\n            result = engine.evaluate_dataset(\n                root=PROJECT_ROOT,\n                dataset=item,\n                ratio=ratio,\n                output_root=out_dir,\n                seed=final_seed,\n                v4_epochs=final_epochs,\n                train_if_missing=train_if_missing,\n            )\n            result["dataset_config"] = cfg\n            results.append(result)\n        except Exception as exc:\n            print(f"[{item}] FAILED: {exc}", file=sys.stderr)\n            failures.append({"dataset": item, "error": str(exc)})\n\n    if results:\n        engine.write_global_outputs(out_dir, results)\n\n    if failures:\n        failure_path = out_dir / "failures.json"\n        failure_path.write_text(\n            json.dumps(failures, ensure_ascii=False, indent=2),\n            encoding="utf-8",\n        )\n        print(f"Some datasets failed. See: {failure_path}", file=sys.stderr)\n        raise SystemExit(1)\n\n    print()\n    print("Done.")\n    print(f"Summary: {out_dir / \'summary.md\'}")\n    return results\n',
    "asca_ad/runner.py": 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any\n\nfrom .config import PROJECT_ROOT, load_dataset_config\n\n\n@dataclass(frozen=True)\nclass DatasetRuntime:\n    name: str\n    epochs: int\n    anormly_ratio: float\n    runner: str\n\n    @property\n    def runner_path(self) -> Path:\n        return PROJECT_ROOT / "scripts" / "dataset_runners" / self.runner\n\n\ndef get_dataset_runtime(name: str) -> DatasetRuntime:\n    config = load_dataset_config()\n    key = name.strip().upper()\n    if key not in config:\n        valid = ", ".join(config.keys())\n        raise ValueError(f"Unknown dataset: {key}. Valid datasets: {valid}")\n\n    item: dict[str, Any] = config[key]\n    return DatasetRuntime(\n        name=key,\n        epochs=int(item.get("epochs", 3)),\n        anormly_ratio=float(item["anormly_ratio"]),\n        runner=str(item["runner"]),\n    )\n\n\ndef list_datasets() -> list[str]:\n    return list(load_dataset_config().keys())\n',
    "data_factory/registry.py": 'from __future__ import annotations\n\nfrom asca_ad.config import load_dataset_config\n\n\ndef available_datasets() -> list[str]:\n    return list(load_dataset_config().keys())\n\n\ndef get_dataset_config(name: str) -> dict:\n    config = load_dataset_config()\n    key = name.strip().upper()\n    if key not in config:\n        valid = ", ".join(config.keys())\n        raise ValueError(f"Unknown dataset: {key}. Valid datasets: {valid}")\n    return config[key]\n',
    "scripts/evaluate.py": '#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\n\nfrom asca_ad.evaluator import run_evaluation\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(\n        description="Run ASCA-AD fixed-combined evaluation by dataset name."\n    )\n    parser.add_argument(\n        "dataset",\n        help="Dataset name: SKAB, PUMP, PSM, MSL, SMAP, HAI, SMD, WADI, or all.",\n    )\n    parser.add_argument("--output-dir", default=None)\n    parser.add_argument("--seed", type=int, default=None)\n    parser.add_argument("--epochs", type=int, default=None)\n    parser.add_argument("--train-if-missing", action="store_true")\n    args = parser.parse_args()\n\n    run_evaluation(\n        dataset=args.dataset,\n        output_dir=args.output_dir,\n        seed=args.seed,\n        epochs=args.epochs,\n        train_if_missing=args.train_if_missing,\n    )\n\n\nif __name__ == "__main__":\n    main()\n',
    "asca_ad/__init__.py": '"""ASCA-AD package."""\n\nfrom .evaluator import run_evaluation\n\n__all__ = ["run_evaluation"]\n',
    "docs/stage3_core_backend.md": '# Stage 3 core backend\n\nThe public CLI is now thin:\n\n    python scripts/evaluate.py SKAB\n\nThe evaluation logic lives in:\n\n    asca_ad/evaluator.py\n\nDataset configuration helpers live in:\n\n    asca_ad/config.py\n    asca_ad/runner.py\n    data_factory/registry.py\n\nThe current backend still uses the validated fixed-combined engine:\n\n    scripts/run_all_fixed_combined.py\n    scripts/dataset_runners/\n\nThis is intentional for compatibility. After all datasets are verified, the legacy\ndataset runners can be merged into a fully unified backend.\n',
}


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def main() -> None:
    root = Path.cwd()

    required = [
        root / "scripts" / "run_all_fixed_combined.py",
        root / "scripts" / "dataset_runners",
        root / "configs" / "datasets.json",
        root / "configs" / "experiment.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files/directories:\n" + "\n".join(missing))

    for rel, content in FILES.items():
        write(root / rel, content, executable=(rel == "scripts/evaluate.py"))

    print("Stage 3 files created.")
    print("Next test:")
    print("  python -m py_compile scripts/evaluate.py asca_ad/config.py asca_ad/evaluator.py asca_ad/runner.py data_factory/registry.py")
    print("  python -u scripts/evaluate.py SKAB")


if __name__ == "__main__":
    main()
