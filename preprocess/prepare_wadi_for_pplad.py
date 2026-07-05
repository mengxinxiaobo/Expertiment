#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OFFICIAL_OUTPUTS = {
    "train": "WaDi_train.npy",
    "test": "WaDi_test.npy",
    "label": "WaDi_test_label.npy",
}


def norm_name(value: object) -> str:
    text = str(value).strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def display_name(value: object) -> str:
    text = str(value).strip().replace("\\", "/").split("/")[-1].strip()
    return text or str(value).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_csv_header_row(path: Path, max_lines: int = 300) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for index, line in enumerate(f):
            if index >= max_lines:
                break
            tokens = [norm_name(v) for v in line.rstrip("\r\n").split(",")[:5]]
            if len(tokens) >= 3 and tokens[:3] == ["row", "date", "time"]:
                return index
    raise ValueError(
        f"Could not find a CSV header beginning with Row,Date,Time in {path}."
    )


def read_columns(path: Path, header_row: int) -> list[str]:
    frame = pd.read_csv(
        path,
        skiprows=header_row,
        nrows=0,
        encoding="utf-8-sig",
    )
    return [str(c) for c in frame.columns]


def locate_metadata_columns(columns: list[str]) -> tuple[str, str, str, list[str]]:
    normalized = {norm_name(c): c for c in columns}
    required = {}
    for key in ("row", "date", "time"):
        if key not in normalized:
            raise ValueError(
                f"Required column {key!r} was not found. Columns: {columns[:10]} ..."
            )
        required[key] = normalized[key]

    label_like = []
    for c in columns:
        n = norm_name(c)
        if n in {
            "attack", "label", "attack_label", "attack_lable",
            "normal_attack", "anomaly", "is_attack",
        }:
            label_like.append(c)

    excluded = {required["row"], required["date"], required["time"], *label_like}
    features = [c for c in columns if c not in excluded]
    return required["row"], required["date"], required["time"], features


def iter_csv(
    path: Path,
    header_row: int,
    usecols: list[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        skiprows=header_row,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    )


def parse_timestamps(date_values: pd.Series, time_values: pd.Series) -> pd.Series:
    combined = (
        date_values.astype("string").str.strip()
        + " "
        + time_values.astype("string").str.strip()
    )
    try:
        parsed = pd.to_datetime(combined, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(combined, errors="coerce")
    return parsed


def numeric_block(
    frame: pd.DataFrame,
    feature_columns: list[str],
    source_name: str,
) -> tuple[np.ndarray, int, int]:
    raw = frame[feature_columns]
    numeric = raw.apply(pd.to_numeric, errors="coerce")

    invalid = 0
    for c in feature_columns:
        nonempty = raw[c].notna() & raw[c].astype("string").str.strip().ne("")
        invalid += int((nonempty & numeric[c].isna()).sum())

    array = numeric.to_numpy(dtype=np.float32, copy=True)
    inf_count = int(np.isinf(array).sum())
    if inf_count:
        array[np.isinf(array)] = np.nan

    return array, invalid, inf_count


def scan_csv(
    path: Path,
    header_row: int,
    row_col: str,
    date_col: str,
    time_col: str,
    feature_columns: list[str],
    chunksize: int,
) -> dict:
    rows = 0
    invalid_numeric = 0
    inf_count = 0
    nan_counts = np.zeros(len(feature_columns), dtype=np.int64)
    first_timestamp = None
    last_timestamp = None

    usecols = [row_col, date_col, time_col, *feature_columns]
    for chunk in iter_csv(path, header_row, usecols, chunksize):
        if chunk.empty:
            continue
        values, invalid, inf = numeric_block(chunk, feature_columns, path.name)
        parsed = parse_timestamps(chunk[date_col], chunk[time_col])
        if parsed.isna().any():
            bad = int(parsed.isna().sum())
            example = (
                chunk.loc[parsed.isna(), [date_col, time_col]]
                .head(3)
                .to_dict("records")
            )
            raise ValueError(
                f"{path.name} has {bad} unparseable timestamps. Examples: {example}"
            )

        rows += len(chunk)
        invalid_numeric += invalid
        inf_count += inf
        nan_counts += np.isnan(values).sum(axis=0).astype(np.int64)
        if first_timestamp is None:
            first_timestamp = parsed.iloc[0]
        last_timestamp = parsed.iloc[-1]

    if rows == 0:
        raise ValueError(f"{path} contains no data rows.")
    if invalid_numeric:
        raise ValueError(
            f"{path.name} contains {invalid_numeric} non-empty non-numeric "
            "feature cells. Refusing to guess conversions."
        )

    return {
        "rows": int(rows),
        "invalid_numeric_cells": int(invalid_numeric),
        "inf_cells_replaced_with_nan": int(inf_count),
        "nan_counts": nan_counts,
        "first_timestamp": str(first_timestamp),
        "last_timestamp": str(last_timestamp),
    }


def time_to_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%S")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, (float, int)) and 0 <= float(value) < 1:
        seconds = int(round(float(value) * 24 * 60 * 60)) % (24 * 60 * 60)
        return str(timedelta(seconds=seconds))

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    return text



def collect_test_dates(
    path: Path,
    header_row: int,
    date_col: str,
    time_col: str,
    chunksize: int,
) -> list[date]:
    """Collect the actual calendar dates present in the attack/test CSV."""
    result: set[date] = set()
    for chunk in iter_csv(
        path,
        header_row,
        [date_col, time_col],
        chunksize,
    ):
        parsed = parse_timestamps(chunk[date_col], chunk[time_col])
        if parsed.isna().any():
            raise ValueError(
                f"{path.name} contains unparseable timestamps while "
                "collecting valid test dates."
            )
        result.update(parsed.dt.date.tolist())
    if not result:
        raise ValueError(f"No valid dates were found in {path}.")
    return sorted(result)


def raw_date_candidates(value: object) -> list[date]:
    """
    Return plausible calendar interpretations of a WADI Excel date cell.

    The official WADI attack workbook is known to contain inconsistent Excel
    date values/formatting in some distributions. We therefore keep multiple
    interpretations and later map them only onto dates that actually occur in
    WADI_attackdata.csv.
    """
    candidates: list[date] = []

    def add(candidate: object) -> None:
        if candidate is None or pd.isna(candidate):
            return
        if isinstance(candidate, pd.Timestamp):
            candidate = candidate.date()
        elif isinstance(candidate, datetime):
            candidate = candidate.date()
        if isinstance(candidate, date) and candidate not in candidates:
            candidates.append(candidate)

    if isinstance(value, (pd.Timestamp, datetime, date)):
        add(value)

    if isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    ):
        # Excel serial-date interpretation.
        try:
            add(
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(float(value), unit="D")
            )
        except Exception:
            pass

    text = str(value).strip()
    if text and text.lower() not in {"nan", "nat", "none"}:
        for dayfirst in (False, True):
            try:
                parsed = pd.to_datetime(
                    text,
                    errors="coerce",
                    dayfirst=dayfirst,
                )
            except Exception:
                parsed = pd.NaT
            add(parsed)

        # Generate explicit numeric interpretations so that values such as
        # 9/10/2017 retain both October-9 and September-10 candidates.
        numbers = [int(v) for v in re.findall(r"\d+", text)]
        if len(numbers) >= 3:
            a, b, y = numbers[0], numbers[1], numbers[2]
            if y < 100:
                y += 2000
            for month, day_value in ((a, b), (b, a)):
                try:
                    add(date(y, month, day_value))
                except ValueError:
                    pass

    return candidates


def map_attack_date_to_test(
    value: object,
    valid_test_dates: list[date],
    previous_date: date | None,
) -> tuple[date, dict]:
    """
    Map an inconsistent Excel date onto a date that really exists in the test
    CSV. The mapping uses only the raw date cell and the test CSV calendar,
    never anomaly labels or model scores.
    """
    candidates = raw_date_candidates(value)
    if not candidates:
        raise ValueError(f"Unparseable attack date cell: {value!r}")

    scored: list[tuple[float, date]] = []
    for valid in valid_test_dates:
        best = -1e9
        for candidate in candidates:
            score = 0.0
            if valid == candidate:
                score += 10000.0
            if (
                valid.month == candidate.month
                and valid.day == candidate.day
            ):
                score += 5000.0
            if valid.day == candidate.day:
                score += 1200.0
            if valid.month == candidate.month:
                score += 300.0

            # Compare month/day after replacing the obviously unreliable year.
            try:
                projected = date(valid.year, candidate.month, candidate.day)
                score -= abs((valid - projected).days)
            except ValueError:
                score -= 366.0

            if previous_date is not None and valid < previous_date:
                score -= 150.0
            best = max(best, score)
        scored.append((best, valid))

    scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    best_score, chosen = scored[0]
    tied = [d for score, d in scored if abs(score - best_score) < 1e-9]
    if len(tied) > 1:
        # Prefer chronological continuity when the workbook row is ambiguous.
        if previous_date is not None:
            nondecreasing = [d for d in tied if d >= previous_date]
            if nondecreasing:
                chosen = min(nondecreasing)
            else:
                chosen = min(tied, key=lambda d: abs((d - previous_date).days))
        else:
            chosen = min(tied)

    primary = candidates[0]
    return chosen, {
        "raw_value": str(value),
        "parsed_candidates": [d.isoformat() for d in candidates],
        "chosen_test_date": chosen.isoformat(),
        "corrected": chosen != primary,
    }


def parse_attack_intervals(
    xlsx_path: Path,
    valid_test_dates: list[date],
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], dict]:
    raw = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
    header_candidates = []
    for idx, row in raw.iterrows():
        normalized = [norm_name(v) for v in row.tolist()]
        if "s_no" in normalized or "sno" in normalized or "s_no_" in normalized:
            header_candidates.append(int(idx))
    if not header_candidates:
        # Common official sheet uses literal "S.No".
        for idx, row in raw.iterrows():
            if any(str(v).strip().lower() == "s.no" for v in row.tolist()):
                header_candidates.append(int(idx))
    if not header_candidates:
        raise ValueError(
            f"Could not find the attack table header row (S.No) in {xlsx_path}."
        )

    header_row = header_candidates[0]
    meta = pd.read_excel(
        xlsx_path,
        header=header_row,
        engine="openpyxl",
    )
    meta.columns = [str(c).strip() for c in meta.columns]

    normalized = {norm_name(c): c for c in meta.columns}
    date_col = next((c for n, c in normalized.items() if n == "date"), None)
    start_col = next(
        (c for n, c in normalized.items() if "start" in n and "time" in n),
        None,
    )
    end_col = next(
        (c for n, c in normalized.items() if "end" in n and "time" in n),
        None,
    )
    if not date_col or not start_col or not end_col:
        raise ValueError(
            "Could not identify Date/Start Time/End Time columns in "
            f"{xlsx_path}. Columns: {list(meta.columns)}"
        )

    intervals = []
    rejected_rows = []
    date_mapping = []
    previous_date: date | None = None

    for row_index, row in meta.iterrows():
        date_value = row.get(date_col)
        start_text = time_to_text(row.get(start_col))
        end_text = time_to_text(row.get(end_col))

        if start_text is None or end_text is None or pd.isna(date_value):
            # Ignore explanatory/footer rows, but record partly populated rows.
            if any(pd.notna(row.get(c)) for c in (date_col, start_col, end_col)):
                rejected_rows.append(int(row_index))
            continue

        chosen_date, mapping = map_attack_date_to_test(
            date_value,
            valid_test_dates,
            previous_date,
        )
        previous_date = chosen_date
        mapping["excel_row_index"] = int(row_index)
        date_mapping.append(mapping)

        try:
            start = pd.to_datetime(
                f"{chosen_date.isoformat()} {start_text}",
                errors="raise",
                format="mixed",
            )
            end = pd.to_datetime(
                f"{chosen_date.isoformat()} {end_text}",
                errors="raise",
                format="mixed",
            )
        except TypeError:
            start = pd.to_datetime(
                f"{chosen_date.isoformat()} {start_text}",
                errors="raise",
            )
            end = pd.to_datetime(
                f"{chosen_date.isoformat()} {end_text}",
                errors="raise",
            )

        if end < start:
            end = end + pd.Timedelta(days=1)
        intervals.append((pd.Timestamp(start), pd.Timestamp(end)))

    intervals = sorted(set(intervals), key=lambda pair: pair[0])
    if not intervals:
        raise ValueError(f"No valid attack intervals were parsed from {xlsx_path}.")

    return intervals, {
        "sheet_header_row_zero_based": int(header_row),
        "date_column": date_col,
        "start_column": start_col,
        "end_column": end_col,
        "valid_interval_count": int(len(intervals)),
        "valid_test_dates": [d.isoformat() for d in valid_test_dates],
        "date_mapping": date_mapping,
        "corrected_date_rows": int(
            sum(bool(item["corrected"]) for item in date_mapping)
        ),
        "rejected_partly_populated_rows": rejected_rows,
    }


def write_csv_to_npy(
    path: Path,
    output_path: Path,
    header_row: int,
    row_col: str,
    date_col: str,
    time_col: str,
    feature_columns: list[str],
    total_rows: int,
    chunksize: int,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    label_output_path: Path | None = None,
) -> dict:
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, len(feature_columns)),
    )
    labels = None
    if intervals is not None:
        if label_output_path is None:
            raise ValueError("label_output_path is required for test data.")
        labels = np.lib.format.open_memmap(
            label_output_path,
            mode="w+",
            dtype=np.int64,
            shape=(total_rows,),
        )

    match_counts = np.zeros(len(intervals or []), dtype=np.int64)
    cursor = 0
    usecols = [row_col, date_col, time_col, *feature_columns]

    for chunk in iter_csv(path, header_row, usecols, chunksize):
        if chunk.empty:
            continue
        values, invalid, _ = numeric_block(chunk, feature_columns, path.name)
        if invalid:
            raise ValueError(
                f"{path.name} developed {invalid} invalid feature cells "
                "between scan and write passes."
            )
        end_cursor = cursor + len(chunk)
        output[cursor:end_cursor] = values

        if labels is not None and intervals is not None:
            parsed = parse_timestamps(chunk[date_col], chunk[time_col])
            chunk_labels = np.zeros(len(chunk), dtype=np.int64)
            timestamps = parsed.to_numpy(dtype="datetime64[ns]")
            for index, (start, end) in enumerate(intervals):
                mask = (
                    (timestamps >= np.datetime64(start.to_datetime64()))
                    & (timestamps <= np.datetime64(end.to_datetime64()))
                )
                if mask.any():
                    chunk_labels[mask] = 1
                    match_counts[index] += int(mask.sum())
            labels[cursor:end_cursor] = chunk_labels

        cursor = end_cursor

    output.flush()
    del output
    if labels is not None:
        labels.flush()
        del labels

    if cursor != total_rows:
        raise RuntimeError(
            f"{path.name}: wrote {cursor} rows, expected {total_rows}."
        )

    return {
        "written_rows": int(cursor),
        "attack_interval_match_counts": match_counts.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the official WADI A1 October 2017 CSV files to the "
            "WaDi_*.npy names expected by the PPLAD WaDi loader."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("dataset/WADI")
    )
    parser.add_argument("--train-csv", default="WADI_14days.csv")
    parser.add_argument("--test-csv", default="WADI_attackdata.csv")
    parser.add_argument("--attack-xlsx", default="attack_description.xlsx")
    parser.add_argument("--expected-channels", type=int, default=127)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument(
        "--hash-files", action="store_true",
        help="Compute SHA-256 hashes for large source/output files."
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    train_path = data_dir / args.train_csv
    test_path = data_dir / args.test_csv
    xlsx_path = data_dir / args.attack_xlsx

    for path in (train_path, test_path, xlsx_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_header = find_csv_header_row(train_path)
    test_header = find_csv_header_row(test_path)
    train_columns = read_columns(train_path, train_header)
    test_columns = read_columns(test_path, test_header)

    train_row, train_date, train_time, train_features = locate_metadata_columns(
        train_columns
    )
    test_row, test_date, test_time, test_features = locate_metadata_columns(
        test_columns
    )

    train_display = [display_name(c) for c in train_features]
    test_display = [display_name(c) for c in test_features]

    if train_display != test_display:
        missing = [c for c in train_display if c not in test_display]
        extra = [c for c in test_display if c not in train_display]
        raise ValueError(
            "Train/test feature schemas differ after removing Row/Date/Time. "
            f"Missing in test={missing}, extra in test={extra}, "
            f"same_order={train_display == test_display}"
        )

    if len(train_features) != args.expected_channels:
        raise ValueError(
            f"Expected {args.expected_channels} WADI process columns but found "
            f"{len(train_features)}. Do not silently drop all-NaN WADI columns, "
            "because PPLAD's official WaDi configuration expects 127 inputs. "
            f"Detected features: {train_display}"
        )

    print("Scanning WADI CSV files (first pass)...")
    train_scan = scan_csv(
        train_path, train_header, train_row, train_date, train_time,
        train_features, args.chunksize
    )
    test_scan = scan_csv(
        test_path, test_header, test_row, test_date, test_time,
        test_features, args.chunksize
    )

    valid_test_dates = collect_test_dates(
        test_path,
        test_header,
        test_date,
        test_time,
        args.chunksize,
    )
    print(
        "Test CSV calendar dates: "
        + ", ".join(d.isoformat() for d in valid_test_dates)
    )
    intervals, interval_meta = parse_attack_intervals(
        xlsx_path,
        valid_test_dates,
    )
    print(f"Parsed attack intervals: {len(intervals)}")
    print(
        "Corrected ambiguous Excel date rows: "
        f"{interval_meta['corrected_date_rows']}"
    )

    train_output = data_dir / OFFICIAL_OUTPUTS["train"]
    test_output = data_dir / OFFICIAL_OUTPUTS["test"]
    label_output = data_dir / OFFICIAL_OUTPUTS["label"]

    for path in (train_output, test_output, label_output):
        path.unlink(missing_ok=True)

    try:
        print("Writing train NPY (second pass)...")
        train_write = write_csv_to_npy(
            train_path, train_output, train_header,
            train_row, train_date, train_time, train_features,
            train_scan["rows"], args.chunksize
        )

        print("Writing test NPY and labels (second pass)...")
        test_write = write_csv_to_npy(
            test_path, test_output, test_header,
            test_row, test_date, test_time, test_features,
            test_scan["rows"], args.chunksize,
            intervals=intervals,
            label_output_path=label_output,
        )

        match_counts = np.asarray(
            test_write["attack_interval_match_counts"], dtype=np.int64
        )
        unmatched = np.flatnonzero(match_counts == 0)
        if unmatched.size:
            details = [
                {
                    "index": int(i),
                    "start": str(intervals[i][0]),
                    "end": str(intervals[i][1]),
                }
                for i in unmatched
            ]
            raise RuntimeError(
                "Some attack-description intervals matched no test rows. "
                f"Details: {details}"
            )
    except Exception:
        # Prevent the pipeline from treating incomplete NPY files as valid.
        for path in (train_output, test_output, label_output):
            path.unlink(missing_ok=True)
        raise

    labels = np.load(label_output, mmap_mode="r")
    attack_points = int(labels.sum())
    attack_ratio = float(labels.mean() * 100.0)

    all_nan_columns = [
        train_display[i]
        for i, count in enumerate(train_scan["nan_counts"])
        if int(count) == train_scan["rows"]
    ]

    columns_path = data_dir / "WaDi_columns.json"
    metadata_path = data_dir / "WaDi_preprocess_metadata.json"
    columns_path.write_text(
        json.dumps(train_display, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "dataset": "WADI A1 October 2017",
        "protocol": (
            "official 14-day normal CSV as train; official attack CSV as test; "
            "attack labels generated from attack_description.xlsx; "
            "Row/Date/Time removed; all 127 process columns retained"
        ),
        "data_dir": str(data_dir),
        "train_csv": train_path.name,
        "test_csv": test_path.name,
        "attack_xlsx": xlsx_path.name,
        "train_header_row_zero_based": int(train_header),
        "test_header_row_zero_based": int(test_header),
        "feature_count": int(len(train_display)),
        "feature_columns": train_display,
        "train_shape": [int(train_scan["rows"]), int(len(train_display))],
        "test_shape": [int(test_scan["rows"]), int(len(train_display))],
        "label_shape": [int(test_scan["rows"])],
        "train_first_timestamp": train_scan["first_timestamp"],
        "train_last_timestamp": train_scan["last_timestamp"],
        "test_first_timestamp": test_scan["first_timestamp"],
        "test_last_timestamp": test_scan["last_timestamp"],
        "train_nan_cells": int(train_scan["nan_counts"].sum()),
        "test_nan_cells": int(test_scan["nan_counts"].sum()),
        "all_nan_train_columns": all_nan_columns,
        "attack_interval_parser": interval_meta,
        "attack_intervals": [
            {
                "start": str(start),
                "end": str(end),
                "matched_points": int(match_counts[i]),
            }
            for i, (start, end) in enumerate(intervals)
        ],
        "test_attack_points": attack_points,
        "test_attack_ratio_percent": attack_ratio,
        "output_files": {
            "train": train_output.name,
            "test": test_output.name,
            "label": label_output.name,
        },
        "pplad_loader_behavior": (
            "PPLAD WaDiSegLoader applies np.nan_to_num, fits StandardScaler "
            "on WaDi_train.npy, and transforms WaDi_test.npy."
        ),
    }

    if args.hash_files:
        metadata["sha256"] = {
            train_path.name: sha256(train_path),
            test_path.name: sha256(test_path),
            xlsx_path.name: sha256(xlsx_path),
            train_output.name: sha256(train_output),
            test_output.name: sha256(test_output),
            label_output.name: sha256(label_output),
        }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("WADI preprocessing completed.")
    print(f"features              : {len(train_display)}")
    print(f"train shape           : {tuple(metadata['train_shape'])}")
    print(f"test shape            : {tuple(metadata['test_shape'])}")
    print(f"label shape           : {tuple(metadata['label_shape'])}")
    print(f"attack intervals      : {len(intervals)}")
    print(f"test attack points    : {attack_points}")
    print(f"test attack ratio     : {attack_ratio:.6f}%")
    print(f"train NaN cells       : {metadata['train_nan_cells']}")
    print(f"test NaN cells        : {metadata['test_nan_cells']}")
    print(f"all-NaN train columns : {all_nan_columns}")
    print(f"output directory      : {data_dir}")


if __name__ == "__main__":
    main()
