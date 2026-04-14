from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.mscred import MSCRED
from utils.data import build_dataloaders
from utils.nasa import ensure_dir, parse_int_sequence, prepare_nasa_cache
from utils.pipeline import (
    apply_thresholds,
    collect_scores,
    compute_thresholds,
    run_epoch,
    save_channel_plots,
    save_history_plot,
    save_metrics,
    save_training_history,
    summarize_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MSCRED on the NASA SMAP/MSL anomaly dataset.")
    parser.add_argument("--raw-data-dir", type=str, default="./archive/data/data")
    parser.add_argument("--labels-path", type=str, default="./archive/labeled_anomalies.csv")
    parser.add_argument("--processed-dir", type=str, default="./data/nasa_processed")
    parser.add_argument("--spacecraft", type=str, default="all")
    parser.add_argument("--channel-id", type=str, default=None)
    parser.add_argument("--channel-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--windows", type=str, default="10,30,60")
    parser.add_argument("--history-steps", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--min-validation-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.98)
    parser.add_argument("--threshold-std-factor", type=float, default=1.5)
    parser.add_argument("--score-topk-ratio", type=float, default=0.02)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument(
        "--balance-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance training samples across channels. Disable with --no-balance-channels.",
    )
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/release/nasa_mscred_best.pth")
    parser.add_argument("--metrics-path", type=str, default="./outputs/release/train_metrics.json")
    parser.add_argument("--scores-path", type=str, default="./outputs/release/train_scores.csv")
    parser.add_argument("--history-path", type=str, default="./outputs/release/train_history.json")
    parser.add_argument("--history-plot-path", type=str, default="./outputs/release/train_history.png")
    parser.add_argument("--plots-dir", type=str, default="./outputs/release/channel_plots")
    parser.add_argument("--max-plots", type=int, default=12)
    return parser


def ensure_cache(args: argparse.Namespace) -> dict:
    processed_dir = Path(args.processed_dir)
    manifest_path = processed_dir / "manifest.json"
    if args.rebuild_cache or not manifest_path.exists():
        return prepare_nasa_cache(
            raw_data_dir=args.raw_data_dir,
            labels_path=args.labels_path,
            output_dir=args.processed_dir,
            spacecraft=args.spacecraft,
            channel_id=args.channel_id,
            channel_limit=args.channel_limit,
            overwrite=args.rebuild_cache,
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    args.windows = parse_int_sequence(args.windows)

    ensure_dir(Path(args.checkpoint_path).parent)
    ensure_dir(Path(args.metrics_path).parent)
    ensure_dir(Path(args.scores_path).parent)
    ensure_dir(Path(args.history_path).parent)
    ensure_dir(Path(args.history_plot_path).parent)

    manifest = ensure_cache(args)
    dataloaders, metadata = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        windows=args.windows,
        history_steps=args.history_steps,
        stride=args.stride,
        validation_ratio=args.validation_ratio,
        min_validation_samples=args.min_validation_samples,
        num_workers=args.num_workers,
        balance_channels=args.balance_channels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSCRED(input_channels=len(args.windows)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None
    best_epoch = 0
    training_history: list[dict[str, float | int | bool]] = []

    print(f"Training on {device} with dataset sizes: {metadata['dataset_sizes']}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        _, val_loss = collect_scores(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            topk_ratio=args.score_topk_ratio,
        )
        improved = val_loss < best_val_loss
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": bool(improved),
            }
        )
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_val={min(best_val_loss, val_loss):.6f}"
        )

        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state_dict)

    val_scores, val_loss = collect_scores(
        model=model,
        dataloader=dataloaders["val"],
        device=device,
        topk_ratio=args.score_topk_ratio,
    )
    thresholds = compute_thresholds(
        validation_scores=val_scores,
        quantile=args.threshold_quantile,
        std_factor=args.threshold_std_factor,
    )

    test_scores, test_loss = collect_scores(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        topk_ratio=args.score_topk_ratio,
    )
    scored_test = apply_thresholds(
        scores=test_scores,
        thresholds=thresholds,
        smooth_window=args.smooth_window,
    )
    metrics = summarize_metrics(scored_test)
    metrics["losses"] = {"validation": val_loss, "test": test_loss}
    metrics["dataset"] = {
        "num_channels": manifest["num_channels"],
        "max_sensors": manifest["max_sensors"],
        "windows": list(args.windows),
        "history_steps": args.history_steps,
        "stride": args.stride,
        "dataset_sizes": metadata["dataset_sizes"],
    }
    metrics["training"] = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_completed": len(training_history),
        "history": training_history,
    }

    save_metrics(metrics, args.metrics_path)
    save_training_history(training_history, args.history_path)
    save_history_plot(training_history, args.history_plot_path)
    scored_test.to_csv(args.scores_path, index=False)
    save_channel_plots(scored_test, args.plots_dir, max_channels=args.max_plots)

    checkpoint_payload = {
        "model_state": best_state_dict,
        "thresholds": thresholds,
        "config": vars(args),
        "metadata": metadata,
        "manifest": manifest,
        "training_history": training_history,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }
    torch.save(checkpoint_payload, args.checkpoint_path)

    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(
        "Global metrics: "
        f"raw_f1={metrics['global'].get('f1', float('nan')):.4f}, "
        f"adjusted_f1={metrics['global_adjusted'].get('f1', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
