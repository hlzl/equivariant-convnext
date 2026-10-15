import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


SOURCE_ROOTS = {"conf", "scripts", "src", "tests"}
SOURCE_FILES = {"AGENTS.md", "README.md", "pyproject.toml", "uv.lock"}


def load_run(
    run_dir: Path,
    device: torch.device,
    checkpoint: str = "best",
    data_root: Path | None = None,
    weights: str = "model",
) -> tuple[DictConfig, torch.nn.Module, dict[str, Any]]:
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    cfg = OmegaConf.load(config_path)
    cfg.data.root = str(
        (data_root or Path(os.environ.get("DATA_ROOT", "~/Data"))).expanduser()
    )
    options = {"pretrained": False} if "pretrained" in cfg.model.module else {}
    model = hydra.utils.instantiate(cfg.model.module, **options).to(device)
    record = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(record[weights], strict=True)
    model.eval()
    return (
        cfg,
        model,
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "config_sha256": file_sha256(config_path),
            "weights": weights,
            "epoch": int(record["epoch"]),
            "data_root": cfg.data.root,
        },
    )


def prepare_analysis(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patch = source_patch() + "\n"
    (output_dir / "source.patch").write_text(patch)
    (output_dir / "env.json").write_text(
        json.dumps(
            {
                "command": sys.argv,
                "git_commit": git_output("rev-parse", "HEAD"),
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "package_lock_sha256": file_sha256(Path("uv.lock")),
            },
            indent=2,
        )
        + "\n"
    )


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_source_path(path: str) -> bool:
    parts = Path(path).parts
    return path in SOURCE_FILES or bool(parts and parts[0] in SOURCE_ROOTS)


def source_patch() -> str:
    paths = sorted(SOURCE_ROOTS | SOURCE_FILES)
    patches = [git_output("diff", "--binary", "HEAD", "--", *paths)]
    untracked = [
        path
        for path in git_output(
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        if is_source_path(path)
    ]
    for path in untracked:
        result = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "/dev/null", path],
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(result.returncode, result.args)
        patches.append(result.stdout.strip())
    return "\n".join(patch for patch in patches if patch)


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_metadata(
    model: torch.nn.Module,
    cfg: DictConfig,
    initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    device = torch.cuda.current_device() if cuda else None
    metadata = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if cuda else "cpu",
        "package_lock_sha256": file_sha256(Path("uv.lock")),
    }
    if cuda:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["nvidia_driver"] = driver.stdout.splitlines()[0]
    if hasattr(model, "provenance"):
        metadata["model_source"] = model.provenance
    if "SLURM_JOB_ID" in os.environ:
        metadata["slurm"] = {
            name: os.environ[name]
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_ACCOUNT",
            )
            if name in os.environ
        }
    if initialization is not None:
        metadata["initialization"] = initialization
    return metadata


def prepare_run(
    run_dir: Path,
    cfg: DictConfig,
    model: torch.nn.Module,
    initialization: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    (run_dir / "env.json").write_text(
        json.dumps(
            run_metadata(model, cfg, initialization),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run_dir / "source.patch").write_text(source_patch() + "\n")


def prepare_or_resume(
    run_dir: Path,
    cfg: DictConfig,
    model: torch.nn.Module,
    initialization: dict[str, Any] | None = None,
    checkpoint_glob: str = "checkpoints/epoch_[0-9][0-9][0-9].pt",
    allow_empty: bool = False,
) -> bool:
    if not run_dir.exists():
        prepare_run(run_dir, cfg, model, initialization)
        return False

    saved_cfg = OmegaConf.load(run_dir / "config.yaml")
    saved_value = OmegaConf.to_container(saved_cfg, resolve=True)
    current_value = OmegaConf.to_container(cfg, resolve=True)
    if saved_value != current_value:
        raise ValueError(f"resume configuration differs: {run_dir}")
    if (run_dir / "source.patch").read_text() != source_patch() + "\n":
        raise ValueError(f"resume source patch differs: {run_dir}")
    metadata = json.loads((run_dir / "env.json").read_text())
    if metadata["git_commit"] != git_output("rev-parse", "HEAD"):
        raise ValueError(f"resume git commit differs: {run_dir}")
    if metadata.get("initialization") != initialization:
        raise ValueError(f"resume initialization differs: {run_dir}")
    if not list(run_dir.glob(checkpoint_glob)):
        if allow_empty:
            return False
        raise ValueError(f"resume checkpoint is missing: {run_dir}")
    return True
