from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.mscred import MSCRED
from utils.data import build_dataloaders
from utils.nasa import ensure_dir, parse_int_sequence, prepare_nasa_cache
from utils.pipeline import (
    apply_thresholds,
    collect_scores,
    compute_thresholds,
    save_channel_plots,
    save_metrics,
    summarize_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained NASA-adapted MSCRED checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/release/nasa_mscred_best.pth")
    parser.add_argument("--raw-data-dir", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--recompute-thresholds", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--score-topk-ratio", type=float, default=None)
    parser.add_argument("--threshold-quantile", type=float, default=None)
    parser.add_argument("--threshold-std-factor", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--metrics-path", type=str, default="./outputs/release/eval_metrics.json")
    parser.add_argument("--scores-path", type=str, default="./outputs/release/eval_scores.csv")
    parser.add_argument("--plots-dir", type=str, default="./outputs/release/eval_plots")
    parser.add_argument("--max-plots", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})

    raw_data_dir = args.raw_data_dir or checkpoint_config.get("raw_data_dir", "./archive/data/data")
    labels_path = args.labels_path or checkpoint_config.get("labels_path", "./archive/labeled_anomalies.csv")
    processed_dir = args.processed_dir or checkpoint_config.get("processed_dir", "./data/nasa_processed")
    windows = parse_int_sequence(checkpoint_config.get("windows", "10,30,60"))
    history_steps = int(checkpoint_config.get("history_steps", 5))
    stride = int(checkpoint_config.get("stride", 5))
    validation_ratio = float(checkpoint_config.get("validation_ratio", 0.15))
    min_validation_samples = int(checkpoint_config.get("min_validation_samples", 8))
    batch_size = int(args.batch_size or checkpoint_config.get("batch_size", 16))
    num_workers = int(args.num_workers or checkpoint_config.get("num_workers", 0))
    smooth_window = int(args.smooth_window or checkpoint_config.get("smooth_window", 1))
    score_topk_ratio = float(args.score_topk_ratio or checkpoint_config.get("score_topk_ratio", 0.02))
    threshold_quantile = float(args.threshold_quantile or checkpoint_config.get("threshold_quantile", 0.98))
    threshold_std_factor = float(args.threshold_std_factor or checkpoint_config.get("threshold_std_factor", 1.5))
    balance_channels = bool(checkpoint_config.get("balance_channels", False))

    ensure_dir(Path(args.metrics_path).parent)
    ensure_dir(Path(args.scores_path).parent)

    manifest_path = Path(processed_dir) / "manifest.json"
    if args.rebuild_cache or not manifest_path.exists():
        prepare_nasa_cache(
            raw_data_dir=raw_data_dir,
            labels_path=labels_path,
            output_dir=processed_dir,
            spacecraft=checkpoint_config.get("spacecraft", "all"),
            channel_id=checkpoint_config.get("channel_id"),
            channel_limit=checkpoint_config.get("channel_limit"),
            overwrite=args.rebuild_cache,
        )

    dataloaders, metadata = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size,
        windows=windows,
        history_steps=history_steps,
        stride=stride,
        validation_ratio=validation_ratio,
        min_validation_samples=min_validation_samples,
        num_workers=num_workers,
        balance_channels=balance_channels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSCRED(input_channels=len(windows)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if args.recompute_thresholds or "thresholds" not in checkpoint:
        val_scores, val_loss = collect_scores(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            topk_ratio=score_topk_ratio,
        )
        thresholds = compute_thresholds(
            validation_scores=val_scores,
            quantile=threshold_quantile,
            std_factor=threshold_std_factor,
        )
    else:
        val_scores, val_loss = collect_scores(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            topk_ratio=score_topk_ratio,
        )
        thresholds = checkpoint["thresholds"]

    test_scores, test_loss = collect_scores(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        topk_ratio=score_topk_ratio,
    )
    scored_test = apply_thresholds(
        scores=test_scores,
        thresholds=thresholds,
        smooth_window=smooth_window,
    )
    metrics = summarize_metrics(scored_test)
    metrics["losses"] = {"validation": val_loss, "test": test_loss}
    metrics["dataset"] = {
        "windows": list(windows),
        "history_steps": history_steps,
        "stride": stride,
        "dataset_sizes": metadata["dataset_sizes"],
    }

    save_metrics(metrics, args.metrics_path)
    scored_test.to_csv(args.scores_path, index=False)
    save_channel_plots(scored_test, args.plots_dir, max_channels=args.max_plots)
    print(
        "Evaluation complete: "
        f"raw_f1={metrics['global'].get('f1', float('nan')):.4f}, "
        f"adjusted_f1={metrics['global_adjusted'].get('f1', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()











