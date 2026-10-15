from typing import Any

import torch
import torch.nn.functional as F


def amp_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unknown AMP precision: {precision}")


def add_confusion(
    confusion: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    num_classes = confusion.shape[0]
    bins = torch.bincount(
        targets * num_classes + predictions,
        minlength=num_classes * num_classes,
    )
    confusion += bins.reshape(num_classes, num_classes)


def macro_accuracy(confusion: torch.Tensor) -> float:
    counts = confusion.sum(dim=1)
    per_class = confusion.diag() / counts.clamp_min(1)
    return float(per_class[counts > 0].mean())


def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    precision: str,
    num_classes: int,
) -> dict[str, float | int]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    loss_sum = 0.0
    sample_count = 0
    use_amp = device.type == "cuda" and precision != "fp32"
    dtype = amp_dtype(precision) if use_amp else None

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().flatten()
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.argmax(dim=1)
            add_confusion(confusion, labels, predictions)
            loss_sum += float(loss)
            sample_count += labels.numel()
    return {
        "loss": loss_sum / sample_count,
        "accuracy": float(confusion.diag().sum() / sample_count),
        "macro_accuracy": macro_accuracy(confusion),
        "samples": sample_count,
    }
