from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_SIGNATURE_WINDOWS = (10, 30, 60)


def parse_int_sequence(value: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(int(item) for item in parts)
    return tuple(int(item) for item in value)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_anomaly_sequences(raw_value: str) -> list[list[int]]:
    sequences = ast.literal_eval(raw_value)
    return [[int(start), int(end)] for start, end in sequences]


def merge_intervals(intervals: Iterable[Sequence[int]]) -> list[list[int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def load_nasa_labels(
    labels_path: str | Path,
    spacecraft: str = "all",
    channel_id: str | None = None,
    channel_limit: int | None = None,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    labels["anomaly_sequences"] = labels["anomaly_sequences"].map(parse_anomaly_sequences)

    grouped = (
        labels.groupby("chan_id", as_index=False)
        .agg(
            spacecraft=("spacecraft", "first"),
            anomaly_sequences=("anomaly_sequences", lambda rows: merge_intervals(seq for row in rows for seq in row)),
            num_values=("num_values", "max"),
        )
        .sort_values("chan_id")
        .reset_index(drop=True)
    )

    if spacecraft.lower() != "all":
        grouped = grouped[grouped["spacecraft"].str.upper() == spacecraft.upper()]

    if channel_id is not None:
        grouped = grouped[grouped["chan_id"] == channel_id]

    grouped = grouped.reset_index(drop=True)
    if channel_limit is not None:
        grouped = grouped.head(channel_limit)
    return grouped


def fit_minmax(train_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_min = train_array.min(axis=0)
    feature_max = train_array.max(axis=0)
    scale = feature_max - feature_min
    scale[scale < 1e-6] = 1.0
    return feature_min.astype(np.float32), scale.astype(np.float32)


def transform_minmax(values: np.ndarray, feature_min: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - feature_min) / scale).astype(np.float32)


def build_point_labels(length: int, anomaly_sequences: Sequence[Sequence[int]]) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int64)
    for start, end in anomaly_sequences:
        left = max(0, int(start))
        right = min(length, int(end))
        if right > left:
            labels[left:right] = 1
    return labels


def prepare_nasa_cache(
    raw_data_dir: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    spacecraft: str = "all",
    channel_id: str | None = None,
    channel_limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    data_root = Path(raw_data_dir)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    cache_dir = ensure_dir(output_dir)

    if overwrite:
        for cache_file in cache_dir.glob("*.npz"):
            cache_file.unlink()

    labels = load_nasa_labels(
        labels_path=labels_path,
        spacecraft=spacecraft,
        channel_id=channel_id,
        channel_limit=channel_limit,
    )

    manifest_records: list[dict] = []
    max_sensors = 0

    for row in labels.itertuples(index=False):
        train_path = train_dir / f"{row.chan_id}.npy"
        test_path = test_dir / f"{row.chan_id}.npy"
        if not train_path.exists() or not test_path.exists():
            continue

        train_array = np.load(train_path).astype(np.float32)
        test_array = np.load(test_path).astype(np.float32)

        feature_min, scale = fit_minmax(train_array)
        train_scaled = transform_minmax(train_array, feature_min, scale)
        test_scaled = transform_minmax(test_array, feature_min, scale)
        point_labels = build_point_labels(len(test_scaled), row.anomaly_sequences)

        sensor_count = int(train_scaled.shape[1])
        max_sensors = max(max_sensors, sensor_count)

        cache_path = cache_dir / f"{row.chan_id}.npz"
        np.savez_compressed(
            cache_path,
            train=train_scaled,
            test=test_scaled,
            test_point_labels=point_labels,
            feature_min=feature_min,
            feature_scale=scale,
        )

        manifest_records.append(
            {
                "channel_id": row.chan_id,
                "spacecraft": row.spacecraft,
                "sensor_count": sensor_count,
                "train_length": int(train_scaled.shape[0]),
                "test_length": int(test_scaled.shape[0]),
                "anomaly_sequences": row.anomaly_sequences,
                "num_values": int(row.num_values),
                "cache_file": cache_path.name,
            }
        )

    manifest = {
        "cache_version": 1,
        "raw_data_dir": str(data_root.resolve()),
        "labels_path": str(Path(labels_path).resolve()),
        "spacecraft": spacecraft,
        "channel_id": channel_id,
        "channel_limit": channel_limit,
        "num_channels": len(manifest_records),
        "max_sensors": max_sensors,
        "channels": manifest_records,
    }
    save_json(cache_dir / "manifest.json", manifest)
    return manifest
