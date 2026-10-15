from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize

import src.data.vision as vision
from src.utils.train import (
    DistributedInfo,
    load_downstream_weights,
    run_pretraining,
    run_training,
)


def cifar_source(root, train, transform, download):
    return ImageFolder(
        Path(root) / ("train" if train else "test"),
        Compose([Resize((32, 32)), transform]),
    )


@pytest.fixture
def sample_root(tmp_path, monkeypatch):
    generator = np.random.default_rng(7)
    for split in ("train", "val", "test"):
        for label in ("0", "1"):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for index in range(4):
                pixels = generator.integers(256, size=(96, 96, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    monkeypatch.setattr(vision.datasets, "CIFAR100", cifar_source)
    return tmp_path


def test_dataloading(sample_root) -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("conf").resolve())
    ):
        for name in vision.DATASET_BUILDERS:
            cfg = compose(config_name="config", overrides=[f"data={name}"])
            for rotation in ("none", "c4", "rot30"):
                cfg.data.rot_aug = rotation
                if name in {"cifar100", "pcam"}:
                    train_transform = (
                        vision.cifar_train_transform
                        if name == "cifar100"
                        else vision.pcam_train_transform
                    )
                    transforms = (
                        train_transform(cfg.data),
                        vision.eval_transform(cfg.data),
                    )
                else:
                    transforms = (
                        vision.task_transform(cfg.data, True),
                        vision.task_transform(cfg.data, False),
                    )
                for transform in transforms:
                    source = ImageFolder(sample_root / "train", transform)
                    loader = vision.data_loader(source, 2, 0, True, 7)
                    images, labels = next(iter(loader))
                    size = cfg.data.get("upscale_size") or cfg.data.image_size
                    assert images.shape == (2, 3, size, size)
                    assert torch.isfinite(images).all()
                    assert (
                        labels.dtype == torch.long
                        and labels.min() >= 0
                        and labels.max() < 2
                    )
        cfg = compose(
            config_name="config", overrides=["data=imagenet1k", "train=imagenet"]
        )
    cfg.data.root = str(sample_root)
    cfg.data.batch_size = 2
    cfg.data.workers = 0
    loaders = vision.make_pretrain_loaders(cfg.data, 7, 0, 1)
    for loader in (loaders.train, loaders.val):
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 224, 224)
        assert torch.isfinite(images).all()
        assert labels.dtype == torch.long
    assert loaders.train.dataset.class_to_idx == loaders.val.dataset.class_to_idx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled CUDA training")
@pytest.mark.parametrize(
    "model_name,recipe",
    [
        ("convnext", "scratch"),
        ("convnext_c4", "scratch"),
        ("convnext_c4_inv", "scratch"),
        ("cvt", "scratch"),
        ("revit_c4", "scratch"),
        ("convnext", "imagenet"),
        ("convnext_c4", "imagenet"),
    ],
)
def test_training(model_name, recipe, sample_root, tmp_path) -> None:
    torch.manual_seed(7)
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("conf").resolve())
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                f"train={recipe}",
                f"data={'imagenet1k' if recipe == 'imagenet' else 'cifar100'}",
                "data.num_classes=2",
                "data.batch_size=2",
                "data.workers=0",
                "train.epochs=1",
                "train.warmup_epochs=0",
            ],
        )
    cfg.data.root = str(sample_root)
    model = hydra.utils.instantiate(cfg.model.module)
    before = model.head.weight.detach().clone()
    device = torch.device("cuda")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if recipe == "imagenet":
        cfg.train.effective_batch = 2
        cfg.train.checkpoint_epochs = 1
        loaders = vision.make_pretrain_loaders(cfg.data, cfg.seed, 0, 1)
        summary = run_pretraining(
            model,
            loaders,
            cfg,
            device,
            run_dir,
            DistributedInfo(rank=0, local_rank=0, world_size=1),
            resume=False,
        )
        checkpoint = run_dir / "checkpoints/epoch_001.pt"
        assert summary["complete"] and summary["epochs"] == 1
        assert (run_dir / "raw.pt").is_file() and (run_dir / "ema.pt").is_file()
    else:
        cfg.data.val_size = 4
        loaders = vision.make_loaders(cfg.data, cfg.seed, include_test=True)
        assert not set(loaders.train.dataset.indices) & set(loaders.val.dataset.indices)
        assert len(loaders.train.dataset) + len(loaders.val.dataset) == 8
        summary = run_training(model, loaders, cfg, device, run_dir)
        checkpoint = run_dir / "checkpoints/best.pt"
        assert summary["completed_epochs"] == 1 and not summary["test_evaluated"]
        assert (run_dir / "checkpoints/last.pt").is_file()
    assert summary["compiler"] == "default"
    assert not torch.equal(before, model.head.weight.detach().cpu())
    assert (run_dir / "summary.json").is_file()
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 1

    record = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = hydra.utils.instantiate(cfg.model.module).to(device).eval()
    restored.load_state_dict(record["model"], strict=True)
    images, _ = next(iter(loaders.val))
    model.eval()
    with torch.inference_mode():
        expected = model.forward_embed(images.to(device))
        torch.testing.assert_close(restored.forward_embed(images.to(device)), expected)
    if model_name in {"convnext", "convnext_c4"}:
        cfg.data.num_classes = 3
        target = hydra.utils.instantiate(cfg.model.module).to(device).eval()
        load_downstream_weights(target, checkpoint, "model", 2718, 0.02)
        with torch.inference_mode():
            torch.testing.assert_close(
                target.forward_embed(images.to(device)), expected
            )
            assert target(images.to(device)).shape == (2, 3)
