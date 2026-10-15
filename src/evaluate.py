"""Evaluate selected downstream checkpoints without training."""

import argparse
from pathlib import Path

import torch

from data.vision import make_loaders
from utils.eval import evaluate
from utils.run import load_run, prepare_analysis
from utils.train import seed_everything, write_json


def evaluate_run(
    run_dir: Path, device: torch.device, split: str, data_root: Path | None = None
) -> dict:
    cfg, model, provenance = load_run(run_dir, device, data_root=data_root)
    seed_everything(cfg.seed)
    loaders = make_loaders(cfg.data, cfg.seed, include_test=split == "test")
    return {
        **provenance,
        "split": split,
        "experiment": cfg.experiment,
        "dataset": cfg.data.name,
        "model": cfg.model.name,
        "seed": cfg.seed,
        "rotation": cfg.data.rot_aug,
        "metrics": evaluate(
            model,
            getattr(loaders, split),
            device,
            cfg.train.precision,
            cfg.data.num_classes,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluate"))
    args = parser.parse_args()
    if args.final_test != (args.split == "test"):
        parser.error("test evaluation requires --split test --final-test")
    prepare_analysis(args.output_dir)
    records = [
        evaluate_run(run, torch.device(args.device), args.split, args.data_root)
        for run in args.run_dirs
    ]
    write_json(args.output_dir / "results.json", records)


if __name__ == "__main__":
    main()
