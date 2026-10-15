"""Measure the effect of averaging the four orientation blocks of a linear head."""

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data.vision import make_loaders, sample_ids
from models.convnext_c4 import C4_ORDER, C4Readout
from utils.eval import add_confusion, amp_dtype, macro_accuracy
from utils.run import file_sha256, load_run, prepare_analysis
from utils.train import seed_everything, write_json


def orientation_invariant_weight(weight: torch.Tensor) -> torch.Tensor:
    """Project a classifier on four ordered C4 blocks onto its invariant part."""
    if weight.ndim != 2 or weight.shape[1] % C4_ORDER:
        raise ValueError(
            "C4 orientation projection requires [classes, 4 * channels] head weights"
        )
    blocks = weight.reshape(weight.shape[0], C4_ORDER, -1)
    invariant = blocks.mean(dim=1, keepdim=True)
    return invariant.expand_as(blocks).reshape_as(weight)


def orientation_decisions(counts: torch.Tensor) -> dict[str, Any]:
    """Summarize paired correctness transitions with equal weight per class."""
    names = (
        "both_correct",
        "ordered_only_correct",
        "averaged_only_correct",
        "both_wrong",
    )
    class_samples = counts.sum(dim=1)
    if torch.any(class_samples == 0):
        raise ValueError("orientation decision analysis requires every class")
    class_fractions = counts / class_samples[:, None]
    sample_count = int(counts.sum())
    categories = {
        name: {
            "count": int(counts[:, index].sum()),
            "fraction": float(counts[:, index].sum() / sample_count),
            "macro_fraction": float(class_fractions[:, index].mean()),
        }
        for index, name in enumerate(names)
    }
    return {
        "categories": categories,
        "class_counts": counts.cpu().tolist(),
        "samples": sample_count,
    }


def orientation_head_metrics(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    precision: str,
    original_weight: torch.Tensor,
    invariant_weight: torch.Tensor,
    num_classes: int,
) -> tuple[
    dict[str, float | int],
    dict[str, float | int],
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    """Evaluate ordered and averaged heads on one shared feature pass."""
    model.eval()
    baseline_confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.int64, device=device
    )
    invariant_confusion = torch.zeros_like(baseline_confusion)
    decision_counts = torch.zeros(num_classes, 4, dtype=torch.int64, device=device)
    label_rows = []
    category_rows = []
    sample_count = 0
    use_amp = device.type == "cuda" and precision != "fp32"
    dtype = amp_dtype(precision) if use_amp else None
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().flatten()
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                features = model.forward_embed(images)
                baseline_logits = F.linear(features, original_weight, model.head.bias)
                invariant_logits = F.linear(features, invariant_weight, model.head.bias)
            baseline_predictions = baseline_logits.argmax(dim=1)
            invariant_predictions = invariant_logits.argmax(dim=1)
            add_confusion(baseline_confusion, labels, baseline_predictions)
            add_confusion(invariant_confusion, labels, invariant_predictions)
            baseline_correct = baseline_predictions.eq(labels)
            invariant_correct = invariant_predictions.eq(labels)
            categories = torch.full_like(labels, 3)
            categories[baseline_correct & invariant_correct] = 0
            categories[baseline_correct & ~invariant_correct] = 1
            categories[~baseline_correct & invariant_correct] = 2
            decision_counts += torch.bincount(
                labels * 4 + categories,
                minlength=num_classes * 4,
            ).reshape(num_classes, 4)
            label_rows.append(labels.cpu())
            category_rows.append(categories.to(torch.uint8).cpu())
            sample_count += labels.numel()
    baseline = {
        "accuracy": float(baseline_confusion.diag().sum() / sample_count),
        "macro_accuracy": macro_accuracy(baseline_confusion),
        "samples": sample_count,
    }
    invariant = {
        "accuracy": float(invariant_confusion.diag().sum() / sample_count),
        "macro_accuracy": macro_accuracy(invariant_confusion),
        "samples": sample_count,
    }
    decisions = orientation_decisions(decision_counts)
    raw = {
        "labels": torch.cat(label_rows),
        "categories": torch.cat(category_rows),
    }
    return baseline, invariant, decisions, raw


def head_average_run(
    run_dir: Path, output_dir: Path, device: torch.device, data_root: Path | None = None
) -> dict[str, Any]:
    cfg, model, provenance = load_run(run_dir, device, data_root=data_root)
    if (
        not isinstance(getattr(model, "readout", None), C4Readout)
        or model.readout.kind != "equivariant"
    ):
        raise ValueError(
            "head averaging requires an equivariant C4 readout and a linear head"
        )
    seed_everything(cfg.seed)
    loader = make_loaders(cfg.data, cfg.seed, include_test=False).val
    weight = model.head.weight.detach()
    baseline, averaged, decisions, raw = orientation_head_metrics(
        model,
        loader,
        device,
        cfg.train.precision,
        weight,
        orientation_invariant_weight(weight),
        cfg.data.num_classes,
    )
    target = (
        output_dir / cfg.experiment / cfg.data.name / cfg.model.name / cfg.data.rot_aug
    )
    target.mkdir(parents=True, exist_ok=True)
    raw_path = target / f"seed_{cfg.seed}.pt"
    torch.save({"ids": sample_ids(loader.dataset, "val"), **raw}, raw_path)
    expected = json.loads((run_dir / "summary.json").read_text())[
        "best_val_macro_accuracy"
    ]
    return {
        **provenance,
        "experiment": cfg.experiment,
        "dataset": cfg.data.name,
        "model": cfg.model.name,
        "rotation": cfg.data.rot_aug,
        "seed": cfg.seed,
        "split": "val",
        "baseline": baseline,
        "averaged": averaged,
        "cost_points": 100 * (baseline["macro_accuracy"] - averaged["macro_accuracy"]),
        "baseline_delta": baseline["macro_accuracy"] - expected,
        "decisions": decisions,
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
    }


def summarize_heads(records: list[dict]) -> dict:
    keys = [
        (row["experiment"], row["model"], row["rotation"], row["dataset"], row["seed"])
        for row in records
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("head averaging received a repeated experiment cell")
    datasets = []
    for key in sorted({item[:-1] for item in keys}):
        selected = [
            row for row, cell in zip(records, keys, strict=True) if cell[:-1] == key
        ]
        costs = [row["cost_points"] for row in selected]
        datasets.append(
            {
                "experiment": key[0],
                "model": key[1],
                "rotation": key[2],
                "dataset": key[3],
                "seeds": len(selected),
                "cost_mean": float(np.mean(costs)),
                "cost_sd": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                "categories": {
                    name: float(
                        np.mean(
                            [
                                row["decisions"]["categories"][name]["macro_fraction"]
                                for row in selected
                            ]
                        )
                    )
                    for name in selected[0]["decisions"]["categories"]
                },
            }
        )
    panels = []
    for key in sorted({item[:3] for item in keys}):
        selected = [
            row
            for row in datasets
            if (row["experiment"], row["model"], row["rotation"]) == key
        ]
        panels.append(
            {
                "experiment": key[0],
                "model": key[1],
                "rotation": key[2],
                "datasets": len(selected),
                "cost_mean": float(np.mean([row["cost_mean"] for row in selected])),
            }
        )
    changes = []
    for model, rotation in sorted({item[1:3] for item in keys}):
        scratch = {
            row["dataset"]: row["cost_mean"]
            for row in datasets
            if (row["experiment"], row["model"], row["rotation"])
            == ("scratch", model, rotation)
        }
        finetune = {
            row["dataset"]: row["cost_mean"]
            for row in datasets
            if (row["experiment"], row["model"], row["rotation"])
            == ("finetune", model, rotation)
        }
        if scratch and scratch.keys() == finetune.keys():
            values = np.array(
                [finetune[name] - scratch[name] for name in sorted(scratch)]
            )
            signs = np.array(list(itertools.product((-1, 1), repeat=len(values))))
            p_value = np.mean(
                np.abs((signs * values).mean(1)) >= abs(values.mean()) - 1e-12
            )
            changes.append(
                {
                    "model": model,
                    "rotation": rotation,
                    "datasets": sorted(scratch),
                    "mean": float(values.mean()),
                    "exact_p": float(p_value),
                }
            )
    return {"datasets": datasets, "panels": panels, "finetune_minus_scratch": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/head_average"))
    args = parser.parse_args()
    prepare_analysis(args.output_dir)
    records = [
        head_average_run(
            run, args.output_dir, torch.device(args.device), args.data_root
        )
        for run in args.run_dirs
    ]
    write_json(args.output_dir / "records.json", records)
    write_json(args.output_dir / "summary.json", summarize_heads(records))
    with (args.output_dir / "table.csv").open("w", newline="") as output:
        fields = ("experiment", "dataset", "model", "rotation", "seed", "cost_points")
        writer = csv.DictWriter(output, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
