import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from data import make_loaders, make_pretrain_loaders
from utils.run import prepare_or_resume, prepare_run
from utils.train import (
    checkpoint_paths,
    close_distributed,
    distributed_barrier,
    load_downstream_weights,
    run_pretraining,
    run_training,
    seed_everything,
    setup_distributed,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.data.name == "imagenet1k":
        if cfg.train.mode != "scratch" or cfg.model.name not in {
            "convnext",
            "convnext_c4",
        }:
            raise ValueError(
                "ImageNet training requires convnext or convnext_c4 from scratch"
            )
        info, device = setup_distributed(cfg.device)
        try:
            seed_everything(cfg.seed)
            torch.backends.cudnn.benchmark = True
            run_dir = Path(os.environ.get("LOCAL_RUN_DIR", cfg.run_dir))
            model = hydra.utils.instantiate(cfg.model.module)
            if info.main:
                prepare_or_resume(run_dir, cfg, model)
            distributed_barrier(info)
            resume = bool(checkpoint_paths(run_dir))
            loaders = make_pretrain_loaders(
                cfg.data,
                cfg.seed,
                info.rank,
                info.world_size,
            )
            summary = run_pretraining(
                model,
                loaders,
                cfg,
                device,
                run_dir,
                info,
                resume,
            )
            if info.main:
                print(json.dumps(summary, indent=2, sort_keys=True))
        finally:
            close_distributed(info)
        return

    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg.device)
    run_dir = Path(os.environ.get("LOCAL_RUN_DIR", cfg.run_dir))
    model = hydra.utils.instantiate(cfg.model.module)
    initialization = None
    if cfg.train.mode == "finetune" and cfg.train.init_checkpoint is not None:
        if cfg.train.init_weights is None:
            raise ValueError("init_weights is required with init_checkpoint")
        checkpoint = Path(os.environ.get("LOCAL_INIT_PATH", cfg.train.init_checkpoint))
        initialization = load_downstream_weights(
            model,
            checkpoint,
            cfg.train.init_weights,
            cfg.train.head_seed,
            cfg.train.head_std,
            Path(cfg.train.init_checkpoint),
        )
    elif cfg.train.mode == "finetune":
        if not cfg.model.module.get("pretrained", False):
            raise ValueError(
                "fine-tuning requires init_checkpoint or a pretrained model"
            )
    elif cfg.train.mode != "scratch":
        raise ValueError(f"unknown training mode: {cfg.train.mode}")
    elif (
        cfg.train.init_weights is not None
        or cfg.train.init_checkpoint is not None
        or cfg.model.module.get("pretrained", False)
    ):
        raise ValueError("pretrained initialization requires train=finetune")

    if (run_dir / "summary.json").exists():
        prepare_or_resume(run_dir, cfg, model, initialization, "checkpoints/last.pt")
        print(f"skip completed run: {run_dir}", flush=True)
        return
    if run_dir.exists():
        resume = prepare_or_resume(
            run_dir,
            cfg,
            model,
            initialization,
            "checkpoints/latest.pt",
            True,
        )
    else:
        prepare_run(run_dir, cfg, model, initialization)
        resume = False
    loaders = make_loaders(cfg.data, cfg.seed, include_test=cfg.train.test_eval)
    summary = run_training(
        model,
        loaders,
        cfg,
        device,
        run_dir,
        resume,
        initialization,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
