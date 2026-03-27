from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import inspect_input
from .predict import predict_to_netcdf
from .train import train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swin3d-insar-inversion")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Inspect a NetCDF/HDF5 input stack")
    inspect_parser.add_argument("--config", required=True)

    train_parser = sub.add_parser("train", help="Train the dual-decoder inversion model")
    train_parser.add_argument("--config", required=True)

    predict_parser = sub.add_parser("predict", help="Run stitched prediction export")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)

    if args.command == "inspect":
        summary = inspect_input(config)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "train":
        summary = train_model(config)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "predict":
        path = predict_to_netcdf(config, args.checkpoint, args.output)
        print(json.dumps({"prediction_path": str(Path(path))}, indent=2))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
