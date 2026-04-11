#!/usr/bin/env python3
# 
# visualizations.py  Andrew Belles  April 10th, 2026 
# 
# Hook to generate UMAP manifold projections for different AE methods
# 

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spiky-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spiky-numba-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "visualizations.yaml"
SPLITS = ("training", "validation", "test")


def log(message: str) -> None:
    print(message, flush=True)


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D UMAP visualizations from manifold parquet outputs.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")

    models = config.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("visualizations config must contain a non-empty 'models' list")

    return {"models": [str(model) for model in models]}


def discover_parquet_groups(data_root: Path, models: list[str]) -> dict[tuple[str, str], dict[str, Path]]:
    groups: dict[tuple[str, str], dict[str, Path]] = {}

    for model in models:
        for split in SPLITS:
            pattern = f"{model}_*_{split}.parquet"
            for path in sorted(data_root.glob(pattern)):
                stem = path.stem
                prefix = f"{model}_"
                suffix = f"_{split}"
                if not stem.startswith(prefix) or not stem.endswith(suffix):
                    continue

                dataset_name = stem[len(prefix) : -len(suffix)]
                groups.setdefault((model, dataset_name), {})[split] = path

    return groups


def read_group_frame(paths_by_split: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for split in SPLITS:
        path = paths_by_split.get(split)
        if path is None or not path.is_file():
            continue

        frame = pd.read_parquet(path).copy()
        frame["split"] = split
        frames.append(frame)

    if not frames:
        raise ValueError("no parquet files were available for visualization")

    combined = pd.concat(frames, ignore_index=True)
    combined["genre_top"] = combined["genre_top"].fillna("unknown")
    return combined


def get_embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith("embedding_"))
    if not columns:
        raise ValueError("no embedding columns found in parquet frame")
    return columns


def compute_umap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    embedding_columns = get_embedding_columns(frame)
    features = frame[embedding_columns].to_numpy(dtype="float32", copy=True)

    if len(frame) < 3:
        raise ValueError("need at least 3 rows to compute UMAP")

    n_neighbors = max(2, min(15, len(frame) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=7,
        n_jobs=1,
    )
    coords = reducer.fit_transform(features)

    result = frame.copy()
    result["umap_x"] = coords[:, 0]
    result["umap_y"] = coords[:, 1]
    return result


def save_umap_figure(frame: pd.DataFrame, model: str, dataset_name: str, output_path: Path) -> Path:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    sns.scatterplot(
        data=frame,
        x="umap_x",
        y="umap_y",
        hue="split",
        ax=axes[0],
        s=14,
        alpha=0.75,
        linewidth=0,
        palette="Set2",
    )
    axes[0].set_title(f"{model} split structure")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    sns.scatterplot(
        data=frame,
        x="umap_x",
        y="umap_y",
        hue="genre_top",
        ax=axes[1],
        s=14,
        alpha=0.75,
        linewidth=0,
        palette="tab20",
    )
    axes[1].set_title(f"{model} genre structure")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")

    for axis in axes:
        if axis.get_legend() is not None:
            sns.move_legend(axis, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    fig.suptitle(f"UMAP | {model} | {dataset_name}", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"

    report(f"START module=visualizations config={args.config}")

    groups = discover_parquet_groups(data_root, config["models"])
    if not groups:
        raise FileNotFoundError(f"no parquet outputs found in {data_root} for models={config['models']}")

    written: list[Path] = []

    for (model, dataset_name), paths_by_split in sorted(groups.items()):
        frame = read_group_frame(paths_by_split)
        umap_frame = compute_umap_frame(frame)
        output_path = image_root / f"{model}_{dataset_name}_umap.png"
        written.append(save_umap_figure(umap_frame, model, dataset_name, output_path))
        log(f"visualized model={model} dataset={dataset_name} path={output_path}")

    report(f"DONE module=visualizations images={len(written)} output_dir={image_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
