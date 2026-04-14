from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.nasa import DEFAULT_SIGNATURE_WINDOWS, load_json, parse_int_sequence


@dataclass
class ChannelSeries:
    channel_id: str
    spacecraft: str
    sensor_count: int
    train: np.ndarray
    test: np.ndarray
    test_point_labels: np.ndarray


def _load_channel_cache(processed_dir: str | Path) -> tuple[dict[str, ChannelSeries], dict]:
    cache_dir = Path(processed_dir)
    manifest = load_json(cache_dir / "manifest.json")
    channels: dict[str, ChannelSeries] = {}
    for channel_record in manifest["channels"]:
        cache_file = cache_dir / channel_record["cache_file"]
        with np.load(cache_file) as cache:
            channels[channel_record["channel_id"]] = ChannelSeries(
                channel_id=channel_record["channel_id"],
                spacecraft=channel_record["spacecraft"],
                sensor_count=int(channel_record["sensor_count"]),
                train=cache["train"].astype(np.float32),
                test=cache["test"].astype(np.float32),
                test_point_labels=cache["test_point_labels"].astype(np.int64),
            )
    return channels, manifest


def _build_anchor_index(
    channels: dict[str, ChannelSeries],
    history_steps: int,
    stride: int,
    validation_ratio: float,
    min_validation_samples: int,
    windows: Iterable[int],
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    max_window = max(windows)
    warmup = max_window - 1 + (history_steps - 1) * stride

    train_entries: list[tuple[str, str, int]] = []
    val_entries: list[tuple[str, str, int]] = []
    test_entries: list[tuple[str, str, int]] = []

    for channel_id, channel in channels.items():
        train_anchors = list(range(warmup, int(channel.train.shape[0]), stride))
        if len(train_anchors) < 2:
            continue

        val_count = max(min_validation_samples, int(len(train_anchors) * validation_ratio))
        val_count = min(val_count, len(train_anchors) - 1)

        train_entries.extend((channel_id, "train", anchor) for anchor in train_anchors[:-val_count])
        val_entries.extend((channel_id, "val", anchor) for anchor in train_anchors[-val_count:])

        test_anchors = list(range(warmup, int(channel.test.shape[0]), stride))
        test_entries.extend((channel_id, "test", anchor) for anchor in test_anchors)

    return train_entries, val_entries, test_entries


class NASASignatureMatrixDataset(Dataset):
    def __init__(
        self,
        channels: dict[str, ChannelSeries],
        entries: list[tuple[str, str, int]],
        windows: Iterable[int],
        history_steps: int,
        stride: int,
        max_sensors: int,
    ) -> None:
        self.channels = channels
        self.entries = entries
        self.windows = tuple(parse_int_sequence(tuple(windows)))
        self.history_steps = int(history_steps)
        self.stride = int(stride)
        self.max_window = max(self.windows)
        self.max_sensors = int(max_sensors)

    def __len__(self) -> int:
        return len(self.entries)

    def _build_mask(self, sensor_count: int) -> np.ndarray:
        mask = np.zeros((len(self.windows), self.max_sensors, self.max_sensors), dtype=np.float32)
        mask[:, :sensor_count, :sensor_count] = 1.0
        return mask

    def _pad_matrix(self, matrix: np.ndarray, sensor_count: int) -> np.ndarray:
        if sensor_count == self.max_sensors:
            return matrix.astype(np.float32)
        padded = np.zeros((self.max_sensors, self.max_sensors), dtype=np.float32)
        padded[:sensor_count, :sensor_count] = matrix
        return padded

    def _signature_matrix(self, sequence: np.ndarray, end_index: int, window_size: int, sensor_count: int) -> np.ndarray:
        window = sequence[end_index - window_size + 1 : end_index + 1]
        matrix = (window.T @ window) / float(window_size)
        return self._pad_matrix(matrix.astype(np.float32), sensor_count)

    def _build_sequence(self, sequence: np.ndarray, anchor: int, sensor_count: int) -> np.ndarray:
        sample_steps: list[np.ndarray] = []
        for step_offset in range(self.history_steps - 1, -1, -1):
            current_anchor = anchor - step_offset * self.stride
            multi_scale = [
                self._signature_matrix(sequence, current_anchor, window_size, sensor_count)
                for window_size in self.windows
            ]
            sample_steps.append(np.stack(multi_scale, axis=0))
        return np.stack(sample_steps, axis=0).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        channel_id, split, anchor = self.entries[index]
        channel = self.channels[channel_id]
        source = channel.train if split in {"train", "val"} else channel.test

        inputs = self._build_sequence(source, anchor, channel.sensor_count)
        target = inputs[-1]
        mask = self._build_mask(channel.sensor_count)

        point_label = 0
        window_label = 0
        if split == "test":
            point_label = int(channel.test_point_labels[anchor])
            left = max(0, anchor - self.max_window + 1)
            window_label = int(channel.test_point_labels[left : anchor + 1].max())

        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "channel_id": channel_id,
            "spacecraft": channel.spacecraft,
            "anchor": torch.tensor(anchor, dtype=torch.long),
            "sensor_count": torch.tensor(channel.sensor_count, dtype=torch.long),
            "point_label": torch.tensor(point_label, dtype=torch.long),
            "window_label": torch.tensor(window_label, dtype=torch.long),
        }


def _build_train_sampler(dataset: NASASignatureMatrixDataset) -> WeightedRandomSampler | None:
    if not dataset.entries:
        return None
    channel_counts = Counter(channel_id for channel_id, _, _ in dataset.entries)
    weights = torch.tensor(
        [1.0 / channel_counts[channel_id] for channel_id, _, _ in dataset.entries],
        dtype=torch.double,
    )
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset.entries), replacement=True)


def build_dataloaders(
    processed_dir: str | Path,
    batch_size: int,
    windows: Iterable[int] = DEFAULT_SIGNATURE_WINDOWS,
    history_steps: int = 5,
    stride: int = 5,
    validation_ratio: float = 0.15,
    min_validation_samples: int = 8,
    num_workers: int = 0,
    balance_channels: bool = True,
) -> tuple[dict[str, DataLoader], dict]:
    channels, manifest = _load_channel_cache(processed_dir)
    parsed_windows = parse_int_sequence(tuple(windows))

    train_entries, val_entries, test_entries = _build_anchor_index(
        channels=channels,
        history_steps=history_steps,
        stride=stride,
        validation_ratio=validation_ratio,
        min_validation_samples=min_validation_samples,
        windows=parsed_windows,
    )

    datasets = {
        "train": NASASignatureMatrixDataset(
            channels=channels,
            entries=train_entries,
            windows=parsed_windows,
            history_steps=history_steps,
            stride=stride,
            max_sensors=manifest["max_sensors"],
        ),
        "val": NASASignatureMatrixDataset(
            channels=channels,
            entries=val_entries,
            windows=parsed_windows,
            history_steps=history_steps,
            stride=stride,
            max_sensors=manifest["max_sensors"],
        ),
        "test": NASASignatureMatrixDataset(
            channels=channels,
            entries=test_entries,
            windows=parsed_windows,
            history_steps=history_steps,
            stride=stride,
            max_sensors=manifest["max_sensors"],
        ),
    }

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_sampler = _build_train_sampler(datasets["train"]) if balance_channels else None
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_kwargs),
    }

    metadata = {
        "manifest": manifest,
        "max_sensors": manifest["max_sensors"],
        "windows": parsed_windows,
        "history_steps": history_steps,
        "stride": stride,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
    }
    return dataloaders, metadata
