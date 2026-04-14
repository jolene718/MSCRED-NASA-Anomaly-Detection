from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nasa import prepare_nasa_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the NASA SMAP/MSL dataset cache used by MSCRED."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="./archive/data/data",
        help="Directory containing the NASA train/test .npy folders.",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="./archive/labeled_anomalies.csv",
        help="Path to labeled_anomalies.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/nasa_processed",
        help="Directory used to store the normalized per-channel cache.",
    )
    parser.add_argument(
        "--spacecraft",
        type=str,
        default="all",
        choices=["all", "SMAP", "MSL", "smap", "msl"],
        help="Optional spacecraft filter.",
    )
    parser.add_argument(
        "--channel-id",
        type=str,
        default=None,
        help="Optional single channel id, for example P-1.",
    )
    parser.add_argument(
        "--channel-limit",
        type=int,
        default=None,
        help="Optional limit used for quick debugging runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the cache files even if they already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_nasa_cache(
        raw_data_dir=args.raw_data_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        spacecraft=args.spacecraft,
        channel_id=args.channel_id,
        channel_limit=args.channel_limit,
        overwrite=args.overwrite,
    )
    print(
        "Prepared NASA cache with "
        f"{manifest['num_channels']} channels; max sensor count = {manifest['max_sensors']}."
    )


if __name__ == "__main__":
    main()
