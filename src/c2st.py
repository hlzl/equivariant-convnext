"""Global and local classifier two-sample tests for dataset orientation."""

import argparse
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.vision import (
    DATASET_BUILDERS,
    data_loader,
    dataset_targets,
    detector_transform,
    evaluation_pool,
    sample_ids,
)
from utils.eval import amp_dtype
from utils.run import file_sha256, prepare_analysis
from utils.train import compile_model, seed_everything, write_json


class DetectorDataset(Dataset):
    def __init__(
        self,
        source: Dataset,
        indices: list[int],
        targets: list[int],
        patch_size: int | None,
        seed: int,
        randomize: bool,
        symmetrize: bool = False,
    ) -> None:
        if len(indices) != len(targets):
            raise ValueError("detector indices and targets must have equal length")
        self.source = source
        self.indices = indices
        self.targets = targets
        self.patch_size = patch_size
        self.randomize = randomize
        self.symmetrize = symmetrize

        generator = torch.Generator().manual_seed(seed)
        count = len(indices)
        self.group_turns = torch.randint(4, (count,), generator=generator)
        self.pre_turns = torch.randint(4, (count,), generator=generator)
        self.crop_draws = torch.rand(count, 2, generator=generator)

    def __len__(self) -> int:
        return len(self.indices)

    def random_turn(self, fixed: torch.Tensor, item: int) -> int:
        if self.randomize:
            return int(torch.randint(4, ()).item())
        return int(fixed[item])

    def crop_image(self, image: torch.Tensor, item: int) -> torch.Tensor:
        if self.patch_size is None:
            return image
        height, width = image.shape[-2:]
        if height != width or self.patch_size > height:
            raise ValueError("local detector requires a valid square image")
        span = height - self.patch_size + 1
        if self.randomize:
            top = int(torch.randint(span, ()).item())
            left = int(torch.randint(span, ()).item())
        else:
            top = min(int(self.crop_draws[item, 0] * span), span - 1)
            left = min(int(self.crop_draws[item, 1] * span), span - 1)
        return image[:, top : top + self.patch_size, left : left + self.patch_size]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int, int]:
        source_index = self.indices[item]
        image, semantic = self.source[source_index]
        target = self.targets[item]

        pre_turn = self.random_turn(self.pre_turns, item)
        if self.symmetrize:
            image = torch.rot90(image, pre_turn, dims=(-2, -1))

        image = self.crop_image(image, item)
        if target == 0:
            turn = self.random_turn(self.group_turns, item)
            image = torch.rot90(image, turn, dims=(-2, -1))
        return image, target, int(semantic), source_index


class C2STDetector(torch.nn.Module):
    """One-hidden-layer classifier for the classifier two-sample test."""

    def __init__(self, in_channels: int, image_size: int) -> None:
        super().__init__()
        input_features = in_channels * image_size**2
        # C2ST-NN from Lopez-Paz and Oquab, ICLR 2017, arXiv:1610.06545:
        # one hidden layer with 20 ReLU units and one sigmoid output. We return
        # [0, z] logits so two-class cross-entropy equals BCEWithLogits(z).
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_features, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logit = self.layers(images).squeeze(1)
        return torch.stack((torch.zeros_like(logit), logit), dim=1)


def stratified_split(
    labels: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    labels = labels.long().flatten()
    classes, counts = labels.unique(return_counts=True, sorted=True)
    val_total = round(labels.numel() * val_fraction)
    raw = val_total * counts.double() / labels.numel()
    quotas = raw.floor().long()
    generator = torch.Generator().manual_seed(seed)
    tie_order = torch.randperm(classes.numel(), generator=generator)
    while int(quotas.sum()) < val_total:
        scores = raw - quotas
        best = scores.max()
        tied = torch.where(scores == best)[0]
        selected = next(index for index in tie_order if index in tied)
        quotas[selected] += 1

    train = []
    validation = []
    for label, quota in zip(classes, quotas, strict=True):
        available = torch.where(labels == label)[0]
        order = torch.randperm(available.numel(), generator=generator)
        validation.extend(available[order[:quota]].tolist())
        train.extend(available[order[quota:]].tolist())
    train = torch.tensor(train)[
        torch.randperm(len(train), generator=generator)
    ].tolist()
    validation = torch.tensor(validation)[
        torch.randperm(len(validation), generator=generator)
    ].tolist()
    return {"train": train, "validation": validation}


def complete_partition(
    labels: torch.Tensor,
    seed: int,
    indices: list[int] | None = None,
) -> dict[str, list[int]]:
    labels = labels.long().flatten()
    candidates = (
        torch.arange(labels.numel()) if indices is None else torch.tensor(indices)
    )
    classes = labels[candidates].unique(sorted=True)
    generator = torch.Generator().manual_seed(seed)
    odd = [
        int(label) for label in classes if int((labels[candidates] == label).sum()) % 2
    ]
    odd_order = torch.randperm(len(odd), generator=generator).tolist()
    extra_observed = {odd[index] for index in odd_order[: (len(odd) + 1) // 2]}
    observed = []
    averaged = []
    for label in classes:
        available = candidates[labels[candidates] == label]
        order = torch.randperm(available.numel(), generator=generator)
        observed_count = available.numel() // 2 + int(int(label) in extra_observed)
        observed.extend(available[order[:observed_count]].tolist())
        averaged.extend(available[order[observed_count:]].tolist())
    observed = torch.tensor(observed)[
        torch.randperm(len(observed), generator=generator)
    ].tolist()
    averaged = torch.tensor(averaged)[
        torch.randperm(len(averaged), generator=generator)
    ].tolist()
    return {"observed": observed, "group_average": averaged}


def detector_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.train()
    use_amp = device.type == "cuda" and precision != "fp32"
    dtype = amp_dtype(precision) if use_amp else None
    loss_sum = 0.0
    correct = 0
    count = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * targets.numel()
        correct += int(logits.argmax(dim=1).eq(targets).sum())
        count += targets.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    result = {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "seconds": seconds,
        "samples_per_second": count / seconds,
    }
    if device.type == "cuda":
        result["memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def train_detector(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    precision: str,
    max_epochs: int,
    patience: int,
    lr: float,
    betas: tuple[float, float],
    compiler: str = "default",
) -> dict[str, Any]:
    model.to(device)
    compile_model(model, compiler)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
    )
    best_epoch = 0
    best_loss = math.inf
    best_state = None
    history = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = detector_epoch(
            model,
            train_loader,
            optimizer,
            device,
            precision,
        )
        val_metrics = evaluate_detector(
            model,
            val_loader,
            device,
            precision,
        )
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        if val_metrics["cross_entropy"] < best_loss:
            best_epoch = epoch
            best_loss = val_metrics["cross_entropy"]
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "validation": history[best_epoch - 1]["validation"],
    }


def detector_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    semantics: torch.Tensor,
) -> dict[str, Any]:
    losses = F.cross_entropy(logits.float(), targets, reduction="none")
    predictions = logits.argmax(dim=1)
    by_class = {}
    for label in semantics.unique().sort().values:
        selected = semantics == label
        by_class[str(int(label))] = {
            "images": int(selected.sum()),
            "accuracy": float(
                predictions[selected].eq(targets[selected]).float().mean()
            ),
            "cross_entropy": float(losses[selected].mean()),
            "observed": int(targets[selected].eq(1).sum()),
            "group_average": int(targets[selected].eq(0).sum()),
        }
    return {
        "images": targets.numel(),
        "accuracy": float(predictions.eq(targets).float().mean()),
        "cross_entropy": float(losses.mean()),
        "observed": int(targets.eq(1).sum()),
        "group_average": int(targets.eq(0).sum()),
        "by_class": by_class,
    }


def evaluate_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    use_amp = device.type == "cuda" and precision != "fp32"
    dtype = amp_dtype(precision) if use_amp else None
    all_logits = []
    all_targets = []
    all_semantics = []
    with torch.inference_mode():
        for images, targets, semantics, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                logits = model(images)
            all_logits.append(logits.float().cpu())
            all_targets.append(targets.long())
            all_semantics.append(semantics.long())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    semantics = torch.cat(all_semantics)
    return detector_metrics(logits, targets, semantics)


def control_images(samples: int = 4096) -> TensorDataset:
    generator = torch.Generator().manual_seed(1729)
    images = 0.05 * torch.randn(samples, 3, 64, 64, generator=generator)
    images[:, :, 2:18, 2:18] += 1.5
    return TensorDataset(images, torch.arange(samples) % 4)


def run_detector(
    source: Dataset,
    name: str,
    scale: str,
    seed: int,
    args: argparse.Namespace,
    symmetrize: bool = False,
) -> dict:
    labels = (
        torch.as_tensor(dataset_targets(source))
        if not isinstance(source, TensorDataset)
        else source.tensors[1]
    )
    partitions = stratified_split(labels, 0.1, 1729)
    train = complete_partition(labels, 1730, partitions["train"])
    val = complete_partition(labels, 1731, partitions["validation"])
    run_seed = (
        1000 * seed + 10 * list(DATASET_BUILDERS).index(name)
        if name in DATASET_BUILDERS
        else 1000 * seed
    )
    run_seed += int(scale == "local")
    seed_everything(run_seed)
    patch = None if scale == "global" else 32
    train_ids = train["observed"] + train["group_average"]
    val_ids = val["observed"] + val["group_average"]
    train_set = DetectorDataset(
        source,
        train_ids,
        [1] * len(train["observed"]) + [0] * len(train["group_average"]),
        patch,
        run_seed,
        True,
        symmetrize,
    )
    val_set = DetectorDataset(
        source,
        val_ids,
        [1] * len(val["observed"]) + [0] * len(val["group_average"]),
        patch,
        run_seed + 10_000,
        False,
        symmetrize,
    )
    train_loader = data_loader(train_set, args.batch_size, args.workers, True, run_seed)
    val_loader = data_loader(val_set, args.batch_size, args.workers, False, run_seed)
    model = C2STDetector(3, 64 if patch is None else patch)
    training = train_detector(
        model,
        train_loader,
        val_loader,
        torch.device(args.device),
        "fp32",
        args.epochs,
        args.patience,
        0.001,
        (0.9, 0.999),
        args.compiler,
    )
    target = args.output_dir / name / scale / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    ids = sample_ids(source, "image")
    split = {
        part: {
            kind: [ids[index] for index in indices] for kind, indices in rows.items()
        }
        for part, rows in (("train", train), ("val", val))
    }
    split_path = target / "split.json"
    write_json(split_path, split)
    torch.save(
        {"model": model.state_dict(), "epoch": training["best_epoch"]},
        target / "best.pt",
    )
    record = {
        "dataset": name,
        "scale": scale,
        "seed": seed,
        "run_seed": run_seed,
        "test_access": False,
        "source_samples": len(source),
        "symmetrized": symmetrize,
        "global_size": 64,
        "patch_size": patch,
        "hidden_features": 20,
        "compiler": args.compiler,
        "precision": "fp32",
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "checkpoint_sha256": file_sha256(target / "best.pt"),
        "split": str(split_path),
        "split_sha256": file_sha256(split_path),
        **training,
    }
    write_json(target / "record.json", record)
    print(
        f"dataset={name} scale={scale} seed={seed} accuracy={training['validation']['accuracy']:.4f}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_BUILDERS),
        default=list(DATASET_BUILDERS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "~/Data"))
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/c2st"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compiler", choices=("default", "eager"), default="default")
    parser.add_argument(
        "--controls",
        action="store_true",
        help="also fit canonical and symmetric synthetic controls",
    )
    args = parser.parse_args()
    prepare_analysis(args.output_dir)
    records = []
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("conf").resolve())
    ):
        for name in args.datasets:
            cfg = compose(config_name="config", overrides=[f"data={name}"])
            cfg.data.root = str(args.data_root.expanduser())
            source = evaluation_pool(cfg.data, detector_transform(64))
            for scale in ("global", "local"):
                for seed in args.seeds:
                    records.append(run_detector(source, name, scale, seed, args))
    if args.controls:
        source = control_images()
        for name, symmetrize in (("canonical", False), ("symmetric", True)):
            for seed in args.seeds:
                records.append(
                    run_detector(source, name, "global", seed, args, symmetrize)
                )
    summary = []
    for name, scale in sorted({(row["dataset"], row["scale"]) for row in records}):
        values = [
            row["validation"]["accuracy"]
            for row in records
            if (row["dataset"], row["scale"]) == (name, scale)
        ]
        summary.append(
            {
                "dataset": name,
                "scale": scale,
                "seeds": len(values),
                "accuracy_mean": float(np.mean(values)),
                "accuracy_sd": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
            }
        )
    write_json(args.output_dir / "records.json", records)
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
