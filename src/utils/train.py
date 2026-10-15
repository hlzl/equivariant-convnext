import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from timm.data import Mixup
from torch.nn.parallel import DistributedDataParallel

from .eval import amp_dtype, evaluate
from .run import file_sha256


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        step = min(self.last_epoch + 1, self.total_steps)
        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(
            1,
            self.total_steps - self.warmup_steps,
        )
        scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs
        ]


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int

    @property
    def main(self) -> bool:
        return self.rank == 0


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_values = self.model.state_dict().values()
        model_values = model.state_dict().values()
        for ema_value, model_value in zip(ema_values, model_values, strict=True):
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(model_value, alpha=1 - self.decay)
            else:
                ema_value.copy_(model_value)


def setup_distributed(device_name: str) -> tuple[DistributedInfo, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device(device_name)
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedInfo(rank, local_rank, world_size), device


def distributed_barrier(info: DistributedInfo) -> None:
    if info.world_size > 1:
        dist.barrier()


def close_distributed(info: DistributedInfo) -> None:
    if info.world_size > 1:
        dist.destroy_process_group()


def compile_model(model: torch.nn.Module, compiler: str) -> None:
    if compiler == "default":
        model.compile(fullgraph=True, dynamic=False)
    elif compiler != "eager":
        raise ValueError(f"unknown compiler mode: {compiler}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("a") as output:
        output.write(json.dumps(data, sort_keys=True) + "\n")


def truncated_normal(
    shape: tuple[int, ...],
    seed: int,
    std: float,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = np.empty(math.prod(shape), dtype=np.float32)
    filled = 0
    while filled < values.size:
        candidates = generator.normal(0.0, std, values.size - filled)
        candidates = candidates[(candidates >= -2.0) & (candidates <= 2.0)]
        count = min(candidates.size, values.size - filled)
        values[filled : filled + count] = candidates[:count]
        filled += count
    return values.reshape(shape)


def initialize_head(model: torch.nn.Module, seed: int, std: float) -> None:
    if not isinstance(model.head, torch.nn.Linear):
        raise ValueError("initialization requires one linear classification head")
    values = truncated_normal(tuple(model.head.weight.shape), seed, std)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(values))
        torch.nn.init.zeros_(model.head.bias)


def head_state_keys(model: torch.nn.Module) -> set[str]:
    head_parameters = {id(parameter) for parameter in model.head.parameters()}
    return {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in head_parameters
    }


def load_weight_record(path: Path) -> dict[str, Any]:
    """Read a training checkpoint or an exported safetensors weight file."""
    if path.suffix != ".safetensors":
        return torch.load(path, map_location="cpu", weights_only=False)
    with safe_open(path, "pt") as weights:
        metadata = weights.metadata()
        state = {key: weights.get_tensor(key) for key in weights.keys()}
    return {metadata["weight_type"]: state, "epoch": int(metadata["epoch"])}


def load_downstream_weights(
    model: torch.nn.Module,
    checkpoint: Path,
    weight_type: str,
    head_seed: int,
    head_std: float,
    record_path: Path | None = None,
) -> dict[str, Any]:
    record = load_weight_record(checkpoint)
    if weight_type not in record:
        raise ValueError(f"checkpoint has no {weight_type} weights: {checkpoint}")

    head_keys = head_state_keys(model)
    source = dict(record[weight_type])
    missing_heads = head_keys - source.keys()
    if missing_heads:
        raise ValueError(f"checkpoint head keys are missing: {sorted(missing_heads)}")
    for key in head_keys:
        del source[key]

    target = model.state_dict()
    encoder_keys = set(target) - head_keys
    missing = encoder_keys - source.keys()
    unexpected = source.keys() - encoder_keys
    if missing or unexpected:
        raise ValueError(
            "checkpoint encoder keys differ: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    shape_errors = [
        key for key in sorted(encoder_keys) if source[key].shape != target[key].shape
    ]
    if shape_errors:
        raise ValueError(f"checkpoint encoder shapes differ: {shape_errors}")

    strict_state = dict(source)
    strict_state.update({key: target[key] for key in head_keys})
    model.load_state_dict(strict_state, strict=True)
    initialize_head(model, head_seed, head_std)
    return {
        "schema": "downstream_init_v1",
        "checkpoint": str(record_path if record_path is not None else checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": int(record["epoch"]),
        "weight_type": weight_type,
        "removed_keys": sorted(head_keys),
        "loaded_keys": sorted(encoder_keys),
        "loaded_key_count": len(encoder_keys),
        "head_seed": head_seed,
        "head_std": head_std,
    }


def adamw_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(("bias", "gamma", "beta")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_mixup(cfg: Any, num_classes: int) -> Mixup:
    return Mixup(
        mixup_alpha=cfg.train.mixup,
        cutmix_alpha=cfg.train.cutmix,
        prob=cfg.train.mix_prob,
        switch_prob=cfg.train.switch_prob,
        mode="batch",
        label_smoothing=cfg.train.label_smoothing,
        num_classes=num_classes,
    )


def soft_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (-targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "checkpoints").glob("epoch_[0-9][0-9][0-9].pt"))


def latest_checkpoint(run_dir: Path) -> Path:
    paths = checkpoint_paths(run_dir)
    if not paths:
        raise ValueError(f"resume checkpoint is missing: {run_dir}")
    return paths[-1]


def atomic_save(value: Any, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp_path)
    temp_path.replace(path)


def trim_metrics(path: Path, epoch: int) -> None:
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    kept = [row for row in rows if row["epoch"] <= epoch]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept))


def move_images(
    images: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    return images.to(device, non_blocking=True, memory_format=memory_format)


def reduce_sum(value: torch.Tensor, info: DistributedInfo) -> torch.Tensor:
    if info.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def reduce_max(value: torch.Tensor, info: DistributedInfo) -> torch.Tensor:
    if info.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def train_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    precision: str,
    label_smoothing: float,
) -> dict[str, float]:
    model.train()
    use_amp = device.type == "cuda" and precision != "fp32"
    dtype = amp_dtype(precision) if use_amp else None
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and precision == "fp16",
    )
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().flatten()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=label_smoothing,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_sum += float(loss.detach()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum())
        sample_count += labels.numel()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    metrics = {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "samples_per_second": sample_count / elapsed,
        "seconds": elapsed,
    }
    if device.type == "cuda":
        metrics["memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        metrics["reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    return metrics


def final_metrics(
    model: torch.nn.Module, loaders: Any, cfg: Any, device: torch.device
) -> dict[str, Any]:
    return {
        "test": evaluate(
            model, loaders.test, device, cfg.train.precision, cfg.data.num_classes
        )
    }


def training_record(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_metrics: dict[str, float] | None,
    best_score: float,
    best_val_metrics: dict[str, float] | None,
    best_epoch: int,
    selection_metric: str,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "val_accuracy": None if val_metrics is None else val_metrics["accuracy"],
        "val_metrics": val_metrics,
        "best_score": best_score,
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
    }


def load_training(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, Any]:
    record = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(record["model"], strict=True)
    optimizer.load_state_dict(record["optimizer"])
    scheduler.load_state_dict(record["scheduler"])
    return record


def evaluation_epochs(cfg: Any) -> set[int]:
    return set(range(cfg.eval_interval, cfg.epochs + 1, cfg.eval_interval)) | {
        cfg.epochs
    }


def run_training(
    model: torch.nn.Module,
    loaders: Any,
    cfg: Any,
    device: torch.device,
    run_dir: Path,
    resume: bool = False,
    initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if not resume:
        metrics_path.unlink(missing_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.betas),
    )
    total_steps = cfg.train.epochs * len(loaders.train)
    eval_epochs = evaluation_epochs(cfg.train)
    scheduler = WarmupCosine(
        optimizer,
        warmup_steps=cfg.train.warmup_epochs * len(loaders.train),
        total_steps=total_steps,
    )
    selection_metric = cfg.train.selection_metric
    best_score = -1.0
    best_val_metrics = None
    best_epoch = -1
    completed_epoch = 0
    last_val_metrics = None
    if resume:
        latest = load_training(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            device,
        )
        completed_epoch = int(latest["epoch"])
        best_score = float(latest["best_score"])
        best_val_metrics = latest["best_val_metrics"]
        best_epoch = int(latest["best_epoch"])
        last_val_metrics = latest["val_metrics"]
        trim_metrics(metrics_path, completed_epoch)
    compile_model(model, cfg.train.compiler)
    wall_start = time.perf_counter()

    for epoch in range(completed_epoch, cfg.train.epochs):
        completed_epoch = epoch + 1
        train_metrics = train_epoch(
            model,
            loaders.train,
            optimizer,
            scheduler,
            device,
            cfg.train.precision,
            cfg.train.label_smoothing,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
        }
        evaluate_epoch = epoch + 1 in eval_epochs
        if not evaluate_epoch:
            append_json(metrics_path, epoch_metrics)
            if completed_epoch == 1:
                atomic_save(
                    training_record(
                        model,
                        optimizer,
                        scheduler,
                        completed_epoch,
                        None,
                        best_score,
                        best_val_metrics,
                        best_epoch,
                        selection_metric,
                    ),
                    checkpoint_dir / "latest.pt",
                )
            print(
                f"epoch={epoch + 1:03d} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"seconds={train_metrics['seconds']:.1f}",
                flush=True,
            )
            continue
        val_metrics = evaluate(
            model,
            loaders.val,
            device,
            cfg.train.precision,
            cfg.data.num_classes,
        )
        epoch_metrics["val"] = val_metrics
        append_json(metrics_path, epoch_metrics)
        print(
            f"epoch={epoch + 1:03d} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro={val_metrics['macro_accuracy']:.4f} "
            f"seconds={train_metrics['seconds']:.1f}",
            flush=True,
        )

        if val_metrics[selection_metric] > best_score:
            best_score = val_metrics[selection_metric]
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            atomic_save(
                training_record(
                    model,
                    optimizer,
                    scheduler,
                    best_epoch,
                    val_metrics,
                    best_score,
                    best_val_metrics,
                    best_epoch,
                    selection_metric,
                ),
                checkpoint_dir / "best.pt",
            )
        last_val_metrics = val_metrics
        if completed_epoch == 1 or evaluate_epoch:
            atomic_save(
                training_record(
                    model,
                    optimizer,
                    scheduler,
                    completed_epoch,
                    last_val_metrics,
                    best_score,
                    best_val_metrics,
                    best_epoch,
                    selection_metric,
                ),
                checkpoint_dir / "latest.pt",
            )

    atomic_save(
        training_record(
            model,
            optimizer,
            scheduler,
            completed_epoch,
            last_val_metrics,
            best_score,
            best_val_metrics,
            best_epoch,
            selection_metric,
        ),
        checkpoint_dir / "last.pt",
    )

    summary = {
        "experiment": cfg.experiment,
        "model": cfg.model.name,
        "dataset": cfg.data.name,
        "seed": cfg.seed,
        "compiler": cfg.train.compiler,
        "optimizer": "adamw",
        "completed_epochs": completed_epoch,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metric": best_score,
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_macro_accuracy": best_val_metrics["macro_accuracy"],
        "test_evaluated": cfg.train.test_eval,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
    }
    if cfg.train.test_eval:
        best = torch.load(
            checkpoint_dir / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best["model"])
        summary.update(final_metrics(model, loaders, cfg, device))
    summary["wall_seconds"] = time.perf_counter() - wall_start
    write_json(run_dir / "summary.json", summary)
    (checkpoint_dir / "latest.pt").unlink(missing_ok=True)
    return summary


def pretrain_epoch(
    train_model: torch.nn.Module,
    raw_model: torch.nn.Module,
    ema: ModelEma,
    loader: Any,
    sampler: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    mixup: Mixup,
    cfg: Any,
    device: torch.device,
    info: DistributedInfo,
    epoch: int,
) -> dict[str, float]:
    sampler.set_epoch(epoch)
    epoch_seed = cfg.seed + epoch * info.world_size + info.rank
    loader.generator.manual_seed(epoch_seed)
    random.seed(epoch_seed)
    np.random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    train_model.train()
    use_amp = device.type == "cuda" and cfg.train.precision != "fp32"
    dtype = amp_dtype(cfg.train.precision) if use_amp else None
    loss_sum = torch.zeros((), device=device)
    sample_count = torch.zeros((), device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    distributed_barrier(info)
    start = time.perf_counter()

    for images, labels in loader:
        images = move_images(images, device, cfg.train.channels_last)
        labels = labels.to(device, non_blocking=True).long().flatten()
        images, targets = mixup(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            logits = train_model(images)
            loss = soft_target_loss(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(raw_model)
        loss_sum += loss.detach() * labels.numel()
        sample_count += labels.numel()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - start, device=device)
    reduce_sum(loss_sum, info)
    reduce_sum(sample_count, info)
    reduce_max(elapsed, info)
    metrics = {
        "loss": float(loss_sum / sample_count),
        "samples_per_second": float(sample_count / elapsed),
        "seconds": float(elapsed),
    }
    if device.type == "cuda":
        memory = torch.tensor(
            torch.cuda.max_memory_allocated(device) / 2**20,
            device=device,
        )
        reserved = torch.tensor(
            torch.cuda.max_memory_reserved(device) / 2**20,
            device=device,
        )
        reduce_max(memory, info)
        reduce_max(reserved, info)
        metrics["memory_mib"] = float(memory)
        metrics["reserved_mib"] = float(reserved)
    return metrics


def pretrain_eval(
    model: torch.nn.Module,
    loader: Any,
    cfg: Any,
    device: torch.device,
    info: DistributedInfo,
) -> dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda" and cfg.train.precision != "fp32"
    dtype = amp_dtype(cfg.train.precision) if use_amp else None
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    top_count = min(5, cfg.data.num_classes)

    with torch.inference_mode():
        for images, labels in loader:
            images = move_images(images, device, cfg.train.channels_last)
            labels = labels.to(device, non_blocking=True).long().flatten()
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.topk(top_count, dim=1).indices
            correct = predictions.eq(labels[:, None])
            totals[0] += loss.double()
            totals[1] += correct[:, 0].sum()
            totals[2] += correct.any(dim=1).sum()
            totals[3] += labels.numel()

    reduce_sum(totals, info)
    return {
        "loss": float(totals[0] / totals[3]),
        "top1": float(totals[1] / totals[3]),
        "top5": float(totals[2] / totals[3]),
        "samples": int(totals[3]),
    }


def pretrain_record(
    raw_model: torch.nn.Module,
    ema: ModelEma,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    cfg: Any,
) -> dict[str, Any]:
    return {
        "model": raw_model.state_dict(),
        "ema": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "seed": cfg.seed,
        "effective_batch": cfg.train.effective_batch,
    }


def load_pretrain(
    path: Path,
    raw_model: torch.nn.Module,
    ema: ModelEma,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    record = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(record["model"], strict=True)
    ema.model.load_state_dict(record["ema"], strict=True)
    optimizer.load_state_dict(record["optimizer"])
    scheduler.load_state_dict(record["scheduler"])
    return int(record["epoch"])


def export_pretrain(
    run_dir: Path,
    raw_model: torch.nn.Module,
    ema: ModelEma,
    epoch: int,
) -> None:
    atomic_save(
        {"model": raw_model.state_dict(), "epoch": epoch, "weight_type": "raw"},
        run_dir / "raw.pt",
    )
    atomic_save(
        {"model": ema.model.state_dict(), "epoch": epoch, "weight_type": "ema"},
        run_dir / "ema.pt",
    )


def run_pretraining(
    model: torch.nn.Module,
    loaders: Any,
    cfg: Any,
    device: torch.device,
    run_dir: Path,
    info: DistributedInfo,
    resume: bool,
) -> dict[str, Any]:
    observed_batch = cfg.data.batch_size * info.world_size
    if observed_batch != cfg.train.effective_batch:
        raise ValueError(
            f"effective batch must be {cfg.train.effective_batch}, got {observed_batch}"
        )
    if cfg.train.epochs % cfg.train.checkpoint_epochs:
        raise ValueError("final epoch must be a checkpoint epoch")

    initialize_head(model, cfg.train.head_seed, cfg.train.head_std)
    if cfg.train.channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    raw_model = model
    ema = ModelEma(raw_model, cfg.train.ema_decay)
    optimizer = torch.optim.AdamW(
        adamw_groups(raw_model, cfg.train.weight_decay),
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
    )
    total_steps = cfg.train.epochs * len(loaders.train)
    scheduler = WarmupCosine(
        optimizer,
        warmup_steps=cfg.train.warmup_epochs * len(loaders.train),
        total_steps=total_steps,
        min_lr=cfg.train.min_lr,
    )
    start_epoch = 0
    metrics_path = run_dir / "metrics.jsonl"
    if resume:
        start_epoch = load_pretrain(
            latest_checkpoint(run_dir),
            raw_model,
            ema,
            optimizer,
            scheduler,
            device,
        )
        if info.main:
            trim_metrics(metrics_path, start_epoch)

    compile_model(raw_model, cfg.train.compiler)
    train_model: torch.nn.Module = raw_model
    if info.world_size > 1:
        train_model = DistributedDataParallel(
            raw_model,
            device_ids=[info.local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    mixup = make_mixup(cfg, cfg.data.num_classes)
    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    checkpoint_dir = run_dir / "checkpoints"
    if info.main:
        checkpoint_dir.mkdir(exist_ok=resume)
    distributed_barrier(info)
    wall_start = time.perf_counter()
    final_row = None

    for epoch_index in range(start_epoch, cfg.train.epochs):
        epoch = epoch_index + 1
        train_metrics = pretrain_epoch(
            train_model,
            raw_model,
            ema,
            loaders.train,
            loaders.train_sampler,
            optimizer,
            scheduler,
            mixup,
            cfg,
            device,
            info,
            epoch_index,
        )
        raw_metrics = pretrain_eval(train_model, loaders.val, cfg, device, info)
        ema_metrics = pretrain_eval(ema.model, loaders.val, cfg, device, info)
        final_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "raw_val": raw_metrics,
            "ema_val": ema_metrics,
        }
        if info.main:
            append_json(metrics_path, final_row)
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
                f"raw_top1={raw_metrics['top1']:.4f} "
                f"ema_top1={ema_metrics['top1']:.4f} "
                f"seconds={train_metrics['seconds']:.1f}",
                flush=True,
            )
        if epoch % cfg.train.checkpoint_epochs == 0:
            if info.main:
                atomic_save(
                    pretrain_record(
                        raw_model,
                        ema,
                        optimizer,
                        scheduler,
                        epoch,
                        cfg,
                    ),
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                )
            distributed_barrier(info)

    if final_row is None:
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        final_row = rows[-1]
    summary = {
        "experiment": cfg.experiment,
        "model": cfg.model.name,
        "dataset": cfg.data.name,
        "seed": cfg.seed,
        "epochs": cfg.train.epochs,
        "checkpoint_epochs": cfg.train.checkpoint_epochs,
        "world_size": info.world_size,
        "per_gpu_batch": cfg.data.batch_size,
        "effective_batch": observed_batch,
        "gradient_accumulation": 1,
        "parameters": parameters,
        "compiler": cfg.train.compiler,
        "channels_last": cfg.train.channels_last,
        "raw_val": final_row["raw_val"],
        "ema_val": final_row["ema_val"],
        "wall_seconds": time.perf_counter() - wall_start,
        "complete": True,
    }
    if info.main:
        export_pretrain(run_dir, raw_model, ema, cfg.train.epochs)
        write_json(run_dir / "summary.json", summary)
    distributed_barrier(info)
    return summary
