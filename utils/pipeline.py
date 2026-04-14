from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction - target) ** 2 * mask
    normalizer = mask.sum().clamp_min(1.0)
    return squared_error.sum() / normalizer


def score_from_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    topk_ratio: float,
) -> torch.Tensor:
    residual = (prediction - target).abs()
    scores = []
    for sample_residual, sample_mask in zip(residual, mask):
        valid_residual = sample_residual[sample_mask > 0]
        k = max(1, int(valid_residual.numel() * topk_ratio))
        scores.append(torch.topk(valid_residual, k=k).values.mean())
    return torch.stack(scores)


def run_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = masked_mse_loss(predictions, targets, mask)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        return 0.0
    return running_loss / total_batches


@torch.no_grad()
def collect_scores(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    topk_ratio: float,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    records: list[dict[str, Any]] = []
    running_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        predictions = model(inputs)
        batch_loss = masked_mse_loss(predictions, targets, mask)
        batch_scores = score_from_residual(predictions, targets, mask, topk_ratio)

        running_loss += float(batch_loss.item())
        total_batches += 1

        anchors = batch["anchor"].tolist()
        sensor_counts = batch["sensor_count"].tolist()
        point_labels = batch["point_label"].tolist()
        window_labels = batch["window_label"].tolist()

        for idx, channel_id in enumerate(batch["channel_id"]):
            records.append(
                {
                    "channel_id": channel_id,
                    "spacecraft": batch["spacecraft"][idx],
                    "anchor": int(anchors[idx]),
                    "sensor_count": int(sensor_counts[idx]),
                    "point_label": int(point_labels[idx]),
                    "window_label": int(window_labels[idx]),
                    "score": float(batch_scores[idx].item()),
                }
            )

    dataframe = pd.DataFrame.from_records(records)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(["channel_id", "anchor"]).reset_index(drop=True)

    average_loss = running_loss / total_batches if total_batches else 0.0
    return dataframe, average_loss


def compute_thresholds(
    validation_scores: pd.DataFrame,
    quantile: float,
    std_factor: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if validation_scores.empty:
        return thresholds

    for channel_id, group in validation_scores.groupby("channel_id"):
        scores = group["score"].to_numpy(dtype=np.float64)
        quantile_threshold = float(np.quantile(scores, quantile))
        z_threshold = float(scores.mean() + std_factor * scores.std())
        thresholds[channel_id] = max(quantile_threshold, z_threshold)
    return thresholds


def smooth_scores(scores: pd.DataFrame, window_size: int) -> pd.DataFrame:
    smoothed = scores.copy()
    if smoothed.empty:
        smoothed["score_smooth"] = []
        return smoothed

    smoothed["score_smooth"] = (
        smoothed.groupby("channel_id")["score"]
        .transform(lambda series: series.rolling(window=window_size, min_periods=1).mean())
    )
    return smoothed


def point_adjust_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    adjusted = predictions.copy()
    anomaly_state = False

    for index in range(len(labels)):
        if labels[index] == 1 and predictions[index] == 1 and not anomaly_state:
            anomaly_state = True
            back_index = index
            while back_index >= 0 and labels[back_index] == 1:
                adjusted[back_index] = 1
                back_index -= 1
        elif labels[index] == 0:
            anomaly_state = False

        if anomaly_state:
            adjusted[index] = 1

    return adjusted


def apply_thresholds(
    scores: pd.DataFrame,
    thresholds: dict[str, float],
    smooth_window: int,
) -> pd.DataFrame:
    scored = smooth_scores(scores, smooth_window)
    if scored.empty:
        scored["threshold"] = []
        scored["prediction"] = []
        scored["prediction_adjusted"] = []
        return scored

    scored["threshold"] = scored["channel_id"].map(thresholds).astype(np.float32)
    scored["prediction"] = (scored["score_smooth"] > scored["threshold"]).astype(np.int64)

    adjusted_predictions = []
    for _, group in scored.groupby("channel_id", sort=False):
        adjusted_predictions.append(
            pd.Series(
                point_adjust_predictions(
                    group["window_label"].to_numpy(dtype=np.int64),
                    group["prediction"].to_numpy(dtype=np.int64),
                ),
                index=group.index,
            )
        )
    scored["prediction_adjusted"] = pd.concat(adjusted_predictions).sort_index().astype(np.int64)
    return scored


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def summarize_metrics(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {"global": {}, "global_adjusted": {}, "per_channel": []}

    global_metrics = classification_metrics(
        labels=scored["window_label"].to_numpy(dtype=np.int64),
        predictions=scored["prediction"].to_numpy(dtype=np.int64),
        scores=scored["score_smooth"].to_numpy(dtype=np.float64),
    )
    global_adjusted = classification_metrics(
        labels=scored["window_label"].to_numpy(dtype=np.int64),
        predictions=scored["prediction_adjusted"].to_numpy(dtype=np.int64),
        scores=scored["score_smooth"].to_numpy(dtype=np.float64),
    )

    per_channel = []
    for channel_id, group in scored.groupby("channel_id"):
        per_channel.append(
            {
                "channel_id": channel_id,
                "spacecraft": group["spacecraft"].iloc[0],
                "num_samples": int(len(group)),
                "num_anomalies": int(group["window_label"].sum()),
                "raw": classification_metrics(
                    labels=group["window_label"].to_numpy(dtype=np.int64),
                    predictions=group["prediction"].to_numpy(dtype=np.int64),
                    scores=group["score_smooth"].to_numpy(dtype=np.float64),
                ),
                "point_adjusted": classification_metrics(
                    labels=group["window_label"].to_numpy(dtype=np.int64),
                    predictions=group["prediction_adjusted"].to_numpy(dtype=np.int64),
                    scores=group["score_smooth"].to_numpy(dtype=np.float64),
                ),
            }
        )

    per_channel.sort(key=lambda item: item["channel_id"])
    return {
        "global": global_metrics,
        "global_adjusted": global_adjusted,
        "per_channel": per_channel,
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def save_training_history(history: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def save_history_plot(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["epoch"], frame["train_loss"], label="train_loss", color="tab:blue", linewidth=1.8)
    axis.plot(frame["epoch"], frame["val_loss"], label="val_loss", color="tab:orange", linewidth=1.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked Reconstruction Loss")
    axis.set_title("MSCRED Training History")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_channel_plots(scored: pd.DataFrame, output_dir: str | Path, max_channels: int | None = None) -> None:
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    channel_groups = list(scored.groupby("channel_id"))
    if max_channels is not None:
        channel_groups = channel_groups[:max_channels]

    for channel_id, group in channel_groups:
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(group["anchor"], group["score_smooth"], label="score", color="black", linewidth=1.5)
        axis.plot(group["anchor"], group["threshold"], label="threshold", color="tab:red", linestyle="--")
        axis.fill_between(
            group["anchor"],
            0,
            group["score_smooth"].max() if len(group) else 1.0,
            where=group["window_label"].to_numpy(dtype=bool),
            color="tab:red",
            alpha=0.12,
            label="ground truth",
        )
        axis.set_title(f"{channel_id} ({group['spacecraft'].iloc[0]})")
        axis.set_xlabel("Anchor Index")
        axis.set_ylabel("Anomaly Score")
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(plot_dir / f"{channel_id}.png", dpi=150)
        plt.close(figure)
