#!/usr/bin/env python3
#
# train_utils.py  Andrew Belles  April 10th, 2026
#
# Shared helpers for manifold training hooks.
#

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_SPLITS = ("training", "validation", "test")


def log(message: str) -> None:
    print(message, flush=True)


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def merge_config(defaults: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(defaults)

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(config_path: Path, defaults: dict) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {config_path}")

    return merge_config(defaults, loaded)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


class MelVectorDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, target_frames: int):
        self.data_dir = data_dir.resolve()
        self.root_dir = self.data_dir.parent
        self.split = split
        self.target_frames = int(target_frames)
        self.manifest_path = self.data_dir / f"manifest_{split}.csv"

        if self.target_frames <= 0:
            raise ValueError("target_frames must be positive")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest for split '{split}': {self.manifest_path}")

        self.frame = pd.read_csv(self.manifest_path)

        if len(self.frame) == 0:
            self.input_dim = 0
            self.n_mels = 0
            self.input_shape = (1, 0, 0)
        else:
            sample = torch.load(self.root_dir / self.frame.iloc[0]["mel_path"], map_location="cpu")
            if sample.ndim != 2:
                raise ValueError(f"expected mel tensor with shape [n_mels, frames], got {tuple(sample.shape)}")
            self.n_mels = int(sample.size(0))
            self.input_dim = int(self.n_mels * self.target_frames)
            self.input_shape = (1, self.n_mels, self.target_frames)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, object]]:
        row = self.frame.iloc[index].to_dict()
        mel_path = self.root_dir / row["mel_path"]
        mel = torch.load(mel_path, map_location="cpu").float()

        if mel.ndim != 2:
            raise ValueError(f"expected mel tensor with shape [n_mels, frames], got {tuple(mel.shape)}")

        pooled = F.adaptive_avg_pool1d(mel.unsqueeze(0), self.target_frames).squeeze(0)
        image = pooled.unsqueeze(0).contiguous()

        row["original_frames"] = int(mel.size(1))
        row["target_frames"] = self.target_frames
        return image, row


def collate_manifest_batch(batch: list[tuple[torch.Tensor, dict[str, object]]]) -> tuple[torch.Tensor, dict[str, list[object]]]:
    inputs = torch.stack([item[0] for item in batch], dim=0)
    keys = batch[0][1].keys()
    metadata = {key: [item[1][key] for item in batch] for key in keys}
    return inputs, metadata


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_manifest_batch,
    )


def build_validation_subset_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    fraction: float,
    min_size: int,
) -> DataLoader | None:
    dataset_size = len(dataset)
    if dataset_size == 0:
        return None

    subset_size = max(1, int(round(dataset_size * fraction)))
    subset_size = max(subset_size, int(min_size))
    subset_size = min(dataset_size, subset_size)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)[:subset_size].tolist()
    subset = Subset(dataset, indices)
    return build_loader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)


@torch.no_grad()
def compute_embedding_change(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    embedding_fn,
    previous_embeddings: torch.Tensor | None,
) -> tuple[torch.Tensor | None, float | None]:
    if loader is None:
        return None, None

    model.eval()
    current_batches: list[torch.Tensor] = []

    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        embeddings = embedding_fn(model, inputs).detach().cpu()
        current_batches.append(embeddings)

    if not current_batches:
        return None, None

    current_embeddings = torch.cat(current_batches, dim=0)
    if previous_embeddings is None:
        return current_embeddings, None

    cosine = F.cosine_similarity(current_embeddings, previous_embeddings, dim=1).mean().item()
    percent_change = max(0.0, (1.0 - cosine) * 100.0)
    return current_embeddings, percent_change


@torch.no_grad()
def export_embeddings_to_parquet(
    model: torch.nn.Module,
    dataset: Dataset,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    batch_infer_fn,
) -> Path:
    model.eval()
    records: list[dict[str, object]] = []

    for inputs, metadata in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        outputs = batch_infer_fn(model, inputs)

        embeddings = outputs["embedding"].detach().cpu().float()
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(1)

        scalar_outputs = {
            name: value.detach().cpu().float().reshape(-1)
            for name, value in outputs.items()
            if name != "embedding"
        }

        for batch_index in range(embeddings.size(0)):
            row = {key: metadata[key][batch_index] for key in metadata}

            for name, values in scalar_outputs.items():
                row[name] = float(values[batch_index].item())

            for dim_index, value in enumerate(embeddings[batch_index].tolist()):
                row[f"embedding_{dim_index:04d}"] = float(value)

            records.append(row)

    frame = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path
