import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_fn


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader | None


@dataclass(frozen=True)
class PretrainLoaders:
    train: DataLoader
    val: DataLoader
    train_sampler: DistributedSampler


class QuarterRotate:
    def __call__(self, image: Any) -> Any:
        turns = int(torch.randint(4, ()).item())
        return transform_fn.rotate(image, 90 * turns)


class ToRgb:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


class FixedSquare:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.width == image.height:
            return transform_fn.resize(
                image,
                [self.size, self.size],
                InterpolationMode.BILINEAR,
                antialias=True,
            )
        image = transform_fn.resize(
            image, self.size, InterpolationMode.BILINEAR, antialias=True
        )
        return transform_fn.center_crop(image, [self.size, self.size])


class IndexedDataset(Dataset):
    def __init__(
        self,
        source: Dataset,
        indices: list[int],
        targets: list[int],
        ids: list[str],
        transform: Any,
    ) -> None:
        if not len(indices) == len(targets) == len(ids):
            raise ValueError("indexed dataset fields must have equal length")
        self.source = source
        self.indices = indices
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        value = self.source[self.indices[item]]
        image = value[0]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[item]


class PathDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        targets: list[int],
        ids: list[str],
        transform: Any,
    ) -> None:
        if not len(paths) == len(targets) == len(ids):
            raise ValueError("path dataset fields must have equal length")
        self.paths = paths
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[item]) as source:
            image = source.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[item]


def rotation_transform(cfg: Any) -> Any | None:
    if cfg.rot_aug == "none":
        return None
    if cfg.rot_aug == "c4":
        return QuarterRotate()
    if cfg.rot_aug == "rot30":
        return transforms.RandomRotation(
            30,
            interpolation=InterpolationMode.BILINEAR,
            fill=tuple(round(255 * value) for value in cfg.mean),
        )
    raise ValueError(f"unknown rotation augmentation: {cfg.rot_aug}")


def stratified_subset(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    generator = np.random.default_rng(seed)
    base, remainder = divmod(limit, len(classes))
    selected = []
    for class_index, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        quota = base + int(class_index < remainder)
        selected.append(indices[:quota])
    selected = np.concatenate(selected)
    generator.shuffle(selected)
    return selected


def tensor_ops(cfg: Any) -> list[Any]:
    return [transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]


def eval_transform(cfg: Any) -> transforms.Compose:
    return transforms.Compose(resize_ops(cfg) + tensor_ops(cfg))


def resize_ops(cfg: Any) -> list[Any]:
    size = getattr(cfg, "upscale_size", None)
    if size is None:
        return []
    return [
        transforms.Resize(
            [size, size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]


def task_transform(cfg: Any, training: bool) -> transforms.Compose:
    operations: list[Any] = [ToRgb()]
    if training:
        if cfg.crop_scale is None:
            operations.extend(
                [
                    transforms.Resize(
                        cfg.resize_size,
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.RandomCrop(cfg.image_size),
                ]
            )
        else:
            operations.append(
                transforms.RandomResizedCrop(
                    cfg.image_size,
                    scale=tuple(cfg.crop_scale),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        if cfg.spatial_aug:
            operations.append(transforms.RandomHorizontalFlip())
        rotation = rotation_transform(cfg)
        if rotation is not None:
            operations.append(rotation)
    else:
        operations.extend(
            [
                transforms.Resize(
                    cfg.resize_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(cfg.image_size),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    return transforms.Compose(operations)


def split_indices(
    strata: np.ndarray,
    seed: int,
) -> dict[str, list[int]]:
    strata = np.asarray(strata).reshape(-1)
    generator = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for label in np.unique(strata):
        indices = np.flatnonzero(strata == label)
        generator.shuffle(indices)
        train_count = int(0.8 * len(indices))
        val_count = int(0.1 * len(indices))
        result["train"].extend(indices[:train_count].tolist())
        result["val"].extend(indices[train_count : train_count + val_count].tolist())
        result["test"].extend(indices[train_count + val_count :].tolist())
    for split in result:
        generator.shuffle(result[split])
    return result


def train_val_indices(
    labels: np.ndarray,
    seed: int,
    val_fraction: float = 0.1,
    val_minimum: int = 0,
) -> dict[str, list[int]]:
    """Stratified train/validation split, one class at a time.

    Validation takes the larger of `val_fraction` of each class and
    `val_minimum` images.
    """
    labels = np.asarray(labels).reshape(-1)
    generator = np.random.default_rng(seed)
    result = {"train": [], "val": []}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        count = max(int(val_fraction * len(indices)), val_minimum)
        result["val"].extend(indices[:count].tolist())
        result["train"].extend(indices[count:].tolist())
    generator.shuffle(result["train"])
    generator.shuffle(result["val"])
    return result


def indexed_split(
    source: Dataset,
    targets: list[int],
    ids: list[str],
    partitions: dict[str, list[int]],
    transforms_by_split: dict[str, Any],
) -> dict[str, IndexedDataset]:
    result = {}
    for split, indices in partitions.items():
        result[split] = IndexedDataset(
            source,
            indices,
            [targets[index] for index in indices],
            [ids[index] for index in indices],
            transforms_by_split[split],
        )
    return result


def path_split(
    paths: list[Path],
    targets: list[int],
    ids: list[str],
    partitions: dict[str, list[int]],
    transforms_by_split: dict[str, Any],
) -> dict[str, PathDataset]:
    result = {}
    for split, indices in partitions.items():
        result[split] = PathDataset(
            [paths[index] for index in indices],
            [targets[index] for index in indices],
            [ids[index] for index in indices],
            transforms_by_split[split],
        )
    return result


def cifar_train_transform(cfg: Any) -> transforms.Compose:
    train_ops = [transforms.RandomCrop(cfg.image_size, padding=4)]
    if cfg.spatial_aug:
        train_ops.append(transforms.RandomHorizontalFlip())
    rotation = rotation_transform(cfg)
    if rotation is not None:
        train_ops.append(rotation)
    train_ops.extend(resize_ops(cfg))
    train_ops.extend(tensor_ops(cfg))
    return transforms.Compose(train_ops)


def cifar_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    train_ops = cifar_train_transform(cfg) if transform is None else transform
    eval_ops = eval_transform(cfg) if transform is None else transform
    train_source = datasets.CIFAR100(
        root=cfg.root,
        train=True,
        transform=train_ops,
        download=True,
    )
    val_source = datasets.CIFAR100(
        root=cfg.root,
        train=True,
        transform=eval_ops,
        download=True,
    )
    test_set = None
    if include_test:
        test_set = datasets.CIFAR100(
            root=cfg.root,
            train=False,
            transform=eval_ops,
            download=True,
        )

    labels = np.asarray(train_source.targets)
    val_indices = stratified_subset(labels, cfg.val_size, seed=1729)
    train_mask = np.ones(len(labels), dtype=bool)
    train_mask[val_indices] = False
    train_indices = np.flatnonzero(train_mask)
    return (
        Subset(train_source, train_indices.tolist()),
        Subset(val_source, val_indices.tolist()),
        test_set,
    )


def pcam_train_transform(cfg: Any) -> transforms.Compose:
    train_ops = []
    if cfg.spatial_aug:
        train_ops.extend(
            [
                transforms.RandomCrop(cfg.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    rotation = rotation_transform(cfg)
    if rotation is not None:
        train_ops.append(rotation)
    train_ops.extend(resize_ops(cfg))
    train_ops.extend(tensor_ops(cfg))
    return transforms.Compose(train_ops)


def pcam_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    train_ops = pcam_train_transform(cfg) if transform is None else transform
    eval_ops = eval_transform(cfg) if transform is None else transform
    train_set = datasets.PCAM(
        root=cfg.root,
        split="train",
        transform=train_ops,
        download=True,
    )
    val_set = datasets.PCAM(
        root=cfg.root,
        split="val",
        transform=eval_ops,
        download=True,
    )
    test_set = (
        datasets.PCAM(
            root=cfg.root,
            split="test",
            transform=eval_ops,
            download=True,
        )
        if include_test
        else None
    )
    return train_set, val_set, test_set


def task_transforms(cfg: Any, transform: Any | None) -> dict[str, Any]:
    if transform is not None:
        return {"train": transform, "val": transform, "test": transform}
    return {
        "train": task_transform(cfg, True),
        "val": task_transform(cfg, False),
        "test": task_transform(cfg, False),
    }


def food_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    split_ops = task_transforms(cfg, transform)
    train_source = datasets.Food101(cfg.root, split="train")
    test_source = datasets.Food101(cfg.root, split="test")
    targets = list(train_source._labels)
    parts = train_val_indices(np.asarray(targets), int(cfg.split_seed))
    root = Path(cfg.root)
    train_ids = [str(path.relative_to(root)) for path in train_source._image_files]
    train_sets = indexed_split(
        train_source,
        targets,
        train_ids,
        parts,
        {"train": split_ops["train"], "val": split_ops["val"]},
    )
    indices = list(range(len(test_source)))
    test_set = IndexedDataset(
        test_source,
        indices,
        list(test_source._labels),
        [str(path.relative_to(root)) for path in test_source._image_files],
        split_ops["test"],
    )
    return train_sets["train"], train_sets["val"], test_set if include_test else None


def caltech_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    source = datasets.Caltech256(cfg.root)
    counts = np.bincount(source.y)
    retained = [
        index
        for index, category in enumerate(source.categories)
        if category != "257.clutter" and counts[index] >= cfg.min_total
    ]
    remap = {label: index for index, label in enumerate(retained)}
    indices = [index for index, label in enumerate(source.y) if label in remap]
    targets = [remap[source.y[index]] for index in indices]
    ids = [
        str(
            Path("caltech256/256_ObjectCategories")
            / source.categories[source.y[index]]
            / f"{source.y[index] + 1:03d}_{source.index[index]:04d}.jpg"
        )
        for index in indices
    ]
    parts = split_indices(np.asarray(targets), int(cfg.split_seed))
    split_ops = task_transforms(cfg, transform)
    local_source = IndexedDataset(source, indices, targets, ids, None)
    local_targets = list(local_source.targets)
    local_ids = list(local_source.ids)
    sets = indexed_split(
        local_source,
        local_targets,
        local_ids,
        parts,
        split_ops,
    )
    return sets["train"], sets["val"], sets["test"] if include_test else None


def minc_paths(root: Path, split: str) -> tuple[list[Path], list[int], list[str]]:
    list_path = root / "labels" / f"{split}1.txt"
    names = list_path.read_text().splitlines()
    classes = sorted({Path(name).parts[-2] for name in names})
    class_map = {name: index for index, name in enumerate(classes)}
    paths = [root / name for name in names]
    targets = [class_map[Path(name).parts[-2]] for name in names]
    return paths, targets, names


def minc_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    root = Path(cfg.root) / cfg.data_dir
    split_ops = task_transforms(cfg, transform)
    result = {}
    for output, source_split in (
        ("train", "train"),
        ("val", "validate"),
        ("test", "test"),
    ):
        paths, targets, ids = minc_paths(root, source_split)
        result[output] = PathDataset(paths, targets, ids, split_ops[output])
    return result["train"], result["val"], result["test"] if include_test else None


def office_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    root = Path(cfg.root) / cfg.data_dir
    domains = sorted(path for path in root.iterdir() if path.is_dir())
    classes = sorted(
        {path.name for domain in domains for path in domain.iterdir() if path.is_dir()}
    )
    class_map = {name: index for index, name in enumerate(classes)}
    paths = [
        path
        for domain in domains
        for class_name in classes
        for path in sorted((domain / class_name).glob("*"))
        if path.is_file()
    ]
    targets = [class_map[path.parent.name] for path in paths]
    ids = [str(path.relative_to(root)) for path in paths]
    groups = [path.parents[1].name for path in paths]
    strata = np.asarray(
        [f"{target}:{group}" for target, group in zip(targets, groups, strict=True)]
    )
    parts = split_indices(strata, int(cfg.split_seed))
    sets = path_split(
        paths,
        targets,
        ids,
        parts,
        task_transforms(cfg, transform),
    )
    return sets["train"], sets["val"], sets["test"] if include_test else None


def aircraft_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    split_ops = task_transforms(cfg, transform)
    # Merge the official train and validation images, then split by class.
    pool_parts = [
        datasets.FGVCAircraft(cfg.root, split=split, annotation_level="variant")
        for split in ("train", "val")
    ]
    pool = ConcatDataset(pool_parts)
    targets = [label for part in pool_parts for label in part._labels]
    ids = [
        str(Path(path).relative_to(cfg.root))
        for part in pool_parts
        for path in part._image_files
    ]
    parts = train_val_indices(
        np.asarray(targets),
        int(cfg.split_seed),
        val_fraction=float(cfg.val_fraction),
        val_minimum=int(cfg.val_minimum),
    )
    train_sets = indexed_split(
        pool,
        targets,
        ids,
        parts,
        {"train": split_ops["train"], "val": split_ops["val"]},
    )
    test_source = datasets.FGVCAircraft(
        cfg.root,
        split="test",
        annotation_level="variant",
    )
    test_set = IndexedDataset(
        test_source,
        list(range(len(test_source))),
        list(test_source._labels),
        [str(Path(path).relative_to(cfg.root)) for path in test_source._image_files],
        split_ops["test"],
    )
    return train_sets["train"], train_sets["val"], test_set if include_test else None


def sun_paths(
    root: Path,
    manifest_file: str,
    split: str,
) -> tuple[list[Path], list[int], list[str]]:
    manifest = json.loads((root / manifest_file).read_text())
    rows = manifest["splits"][split]
    paths = [root / row["path"] for row in rows]
    targets = [int(row["label"]) for row in rows]
    ids = [row["path"] for row in rows]
    return paths, targets, ids


def sun_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    root = Path(cfg.root) / cfg.data_dir
    paths, targets, ids = sun_paths(root, str(cfg.manifest_file), "train")
    parts = train_val_indices(
        np.asarray(targets),
        int(cfg.split_seed),
        val_fraction=float(cfg.val_fraction),
        val_minimum=int(cfg.val_minimum),
    )
    split_ops = task_transforms(cfg, transform)
    train_sets = path_split(
        paths,
        targets,
        ids,
        parts,
        {"train": split_ops["train"], "val": split_ops["val"]},
    )
    test_paths, test_targets, test_ids = sun_paths(root, str(cfg.manifest_file), "test")
    test_set = PathDataset(test_paths, test_targets, test_ids, split_ops["test"])
    return train_sets["train"], train_sets["val"], test_set if include_test else None


def pet_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    split_ops = task_transforms(cfg, transform)
    train_source = datasets.OxfordIIITPet(cfg.root, split="trainval")
    train_targets = list(train_source._labels)
    parts = train_val_indices(
        np.asarray(train_targets),
        int(cfg.split_seed),
        val_fraction=float(cfg.val_fraction),
        val_minimum=int(cfg.val_minimum),
    )
    root = Path(cfg.root)
    train_ids = [str(path.relative_to(root)) for path in train_source._images]
    train_sets = indexed_split(
        train_source,
        train_targets,
        train_ids,
        parts,
        {"train": split_ops["train"], "val": split_ops["val"]},
    )
    test_source = datasets.OxfordIIITPet(cfg.root, split="test")
    test_set = IndexedDataset(
        test_source,
        list(range(len(test_source))),
        list(test_source._labels),
        [str(path.relative_to(root)) for path in test_source._images],
        split_ops["test"],
    )
    return train_sets["train"], train_sets["val"], test_set if include_test else None


def data_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def imagenet_transform(cfg: Any, training: bool) -> Any:
    transform = create_transform(
        input_size=cfg.image_size,
        is_training=training,
        scale=tuple(cfg.crop_scale),
        ratio=tuple(cfg.crop_ratio),
        hflip=cfg.hflip,
        vflip=0.0,
        color_jitter=cfg.color_jitter,
        auto_augment=cfg.rand_aug,
        interpolation=cfg.interpolation,
        mean=tuple(cfg.mean),
        std=tuple(cfg.std),
        crop_pct=cfg.crop_pct,
        re_prob=cfg.random_erase,
        re_mode=cfg.erase_mode,
        re_count=1,
        separate=training,
    )
    if not training:
        return transform
    primary, secondary, final = transform
    rotation = rotation_transform(cfg)
    rotations = [] if rotation is None else [rotation]
    return transforms.Compose(
        primary.transforms + secondary.transforms + rotations + final.transforms
    )


def imagenet_sets(cfg: Any) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    root = Path(cfg.root)
    train_set = datasets.ImageFolder(
        root / "train",
        transform=imagenet_transform(cfg, True),
    )
    val_set = datasets.ImageFolder(
        root / "val",
        transform=imagenet_transform(cfg, False),
    )
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("ImageNet train and validation class mappings differ")
    return train_set, val_set


def pretrain_loader(
    dataset: Dataset,
    cfg: Any,
    sampler: DistributedSampler,
    seed: int,
    drop_last: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=drop_last,
        generator=generator,
    )


def make_pretrain_loaders(
    cfg: Any,
    seed: int,
    rank: int,
    world_size: int,
) -> PretrainLoaders:
    if cfg.name != "imagenet1k":
        raise ValueError(f"unknown pretraining dataset: {cfg.name}")
    train_set, val_set = imagenet_sets(cfg)
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return PretrainLoaders(
        train=pretrain_loader(train_set, cfg, train_sampler, seed + rank, True),
        val=pretrain_loader(val_set, cfg, val_sampler, seed + rank, False),
        train_sampler=train_sampler,
    )


def dataset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, ConcatDataset):
        return np.concatenate(
            [dataset_targets(component) for component in dataset.datasets]
        )
    if isinstance(dataset, Subset):
        targets = dataset_targets(dataset.dataset)
        return targets[np.asarray(dataset.indices)]
    if isinstance(dataset, (IndexedDataset, PathDataset, datasets.CIFAR100)):
        return np.asarray(dataset.targets)
    if isinstance(dataset, datasets.PCAM):
        filename = dataset._FILES[dataset._split]["targets"][0]
        with dataset.h5py.File(dataset._base_folder / filename) as source:
            return np.asarray(source["y"][:, 0, 0, 0])
    raise ValueError(f"target extraction is not defined for {type(dataset).__name__}")


DATASET_BUILDERS = {
    "cifar100": cifar_sets,
    "pcam": pcam_sets,
    "food101": food_sets,
    "caltech256": caltech_sets,
    "minc2500": minc_sets,
    "officehome": office_sets,
    "fgvc_aircraft": aircraft_sets,
    "sun397": sun_sets,
    "oxford_pets": pet_sets,
}


def dataset_sets(
    cfg: Any,
    include_test: bool,
    transform: Any | None = None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    builder = DATASET_BUILDERS.get(cfg.name)
    if builder is None:
        raise ValueError(f"unknown dataset: {cfg.name}")
    return builder(cfg, include_test, transform)


def make_loaders(
    cfg: Any,
    seed: int,
    include_test: bool = True,
) -> DataLoaders:
    train_set, val_set, test_set = dataset_sets(cfg, include_test)

    return DataLoaders(
        train=data_loader(train_set, cfg.batch_size, cfg.workers, True, seed),
        val=data_loader(val_set, cfg.batch_size, cfg.workers, False, seed),
        test=(
            data_loader(test_set, cfg.batch_size, cfg.workers, False, seed)
            if test_set is not None
            else None
        ),
    )


def detector_transform(size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            ToRgb(),
            FixedSquare(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def evaluation_sets(cfg: Any, transform: Any | None = None) -> tuple[Dataset, Dataset]:
    if transform is None:
        transform = (
            eval_transform(cfg)
            if cfg.name in {"cifar100", "pcam"}
            else task_transform(cfg, False)
        )
    train, val, _ = dataset_sets(cfg, include_test=False, transform=transform)
    return train, val


def evaluation_pool(cfg: Any, transform: Any) -> Dataset:
    return ConcatDataset(evaluation_sets(cfg, transform))


def sample_ids(dataset: Dataset, split: str) -> list[str]:
    if isinstance(dataset, ConcatDataset):
        return [
            item
            for index, part in enumerate(dataset.datasets)
            for item in sample_ids(part, f"{split}{index}")
        ]
    if isinstance(dataset, Subset):
        ids = sample_ids(dataset.dataset, split)
        return [ids[int(index)] for index in dataset.indices]
    if isinstance(dataset, (IndexedDataset, PathDataset)):
        return list(dataset.ids)
    if isinstance(dataset, datasets.CIFAR100):
        split = f"cifar100:{'train' if dataset.train else 'test'}"
    elif isinstance(dataset, datasets.PCAM):
        split = f"pcam:{dataset._split}"
    return [f"{split}:{index:08d}" for index in range(len(dataset))]
