#!/usr/bin/env python3
# 
# variational.py  Andrew Belles  April 10th, 2026 
# 
# Variational AE using ResNet-style CNN architecture 
# for encoding/decoding. 
# 

import argparse
from pathlib import Path

import torch

from manifold.common import (
    MuonConfig,
    VariationalAutoEncoder,
    build_single_device_muon_optimizer,
    variational_autoencoder_objective,
)
from manifold.train_utils import (
    DEFAULT_SPLITS,
    MelVectorDataset,
    build_loader,
    build_validation_subset_loader,
    clone_state_dict,
    compute_embedding_change,
    export_embeddings_to_parquet,
    load_config,
    log,
    report,
    resolve_device,
    set_seed,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "variational.yaml"
DEFAULT_CONFIG = {
    "device": "auto",
    "seed": 7,
    "target_frames": 128,
    "batch_size": 32,
    "num_workers": 4,
    "max_epochs": 100,
    "min_epochs": 10,
    "model": {
        "latent_dim": 128,
        "base_channels": 32,
        "channel_multipliers": [1, 2, 4, 8],
        "block_depth": 2,
        "activation": "gelu",
        "dropout": 0.0,
    },
    "objective": {
        "beta": 1.0,
    },
    "optimizer": {
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
        "muon_weight_decay": 0.0,
        "aux_lr": 3e-4,
        "aux_betas": [0.9, 0.95],
        "aux_eps": 1e-10,
        "aux_weight_decay": 0.0,
    },
    "early_stopping": {
        "method": "generalization_loss",
        "generalization_loss_threshold": 2.0,
        "embedding_subset_fraction": 0.15,
        "embedding_subset_min_size": 128,
        "cosine_change_threshold": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the variational autoencoder on mel tensors.")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        required=True,
        help="Path to a *_mel directory, for example preprocess/data/fma_small_mel.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def _kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


def train_epoch(
    model: VariationalAutoEncoder,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_items = 0

    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)

        x_hat, mu, logvar, z = model(inputs)
        loss, recon_loss, kl_loss = variational_autoencoder_objective(x_hat, inputs, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_items += batch_size

    normalizer = max(1, total_items)
    return total_loss / normalizer, total_recon / normalizer, total_kl / normalizer


@torch.no_grad()
def evaluate_epoch(model: VariationalAutoEncoder, loader, device: torch.device, beta: float) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_items = 0

    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        x_hat, mu, logvar, z = model(inputs)
        loss, recon_loss, kl_loss = variational_autoencoder_objective(x_hat, inputs, mu, logvar, beta=beta)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_items += batch_size

    normalizer = max(1, total_items)
    return total_loss / normalizer, total_recon / normalizer, total_kl / normalizer


@torch.no_grad()
def infer_batch(model: VariationalAutoEncoder, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    x_hat, mu, logvar, z = model(inputs)
    reconstruction_mse = (x_hat - inputs).pow(2).flatten(start_dim=1).mean(dim=1)
    kl_divergence = _kl_per_sample(mu, logvar)
    latent_std = torch.exp(0.5 * logvar).mean(dim=1)
    return {
        "embedding": mu,
        "reconstruction_mse": reconstruction_mse,
        "kl_divergence": kl_divergence,
        "latent_std": latent_std,
    }


@torch.no_grad()
def embedding_batch(model: VariationalAutoEncoder, inputs: torch.Tensor) -> torch.Tensor:
    mu, logvar = model.encode(inputs)
    return mu


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))

    data_dir = args.data_dir.expanduser().resolve()
    target_frames = int(config["target_frames"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    beta = float(config["objective"]["beta"])

    report(f"START module=variational data_dir={data_dir} config={args.config}")

    train_dataset = MelVectorDataset(data_dir, "training", target_frames)
    val_dataset = MelVectorDataset(data_dir, "validation", target_frames)
    test_dataset = MelVectorDataset(data_dir, "test", target_frames)

    if train_dataset.input_dim <= 0:
        raise ValueError(f"training split is empty for {data_dir}")

    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, device=device)
    val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)
    subset_loader = build_validation_subset_loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=int(config["seed"]),
        fraction=float(config["early_stopping"]["embedding_subset_fraction"]),
        min_size=int(config["early_stopping"]["embedding_subset_min_size"]),
    )

    model = VariationalAutoEncoder(
        input_shape=train_dataset.input_shape,
        latent_dim=int(config["model"]["latent_dim"]),
        base_channels=int(config["model"]["base_channels"]),
        channel_multipliers=tuple(config["model"]["channel_multipliers"]),
        block_depth=int(config["model"]["block_depth"]),
        activation=str(config["model"]["activation"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    optimizer = build_single_device_muon_optimizer(
        model,
        MuonConfig(
            muon_lr=float(config["optimizer"]["muon_lr"]),
            muon_momentum=float(config["optimizer"]["muon_momentum"]),
            muon_weight_decay=float(config["optimizer"]["muon_weight_decay"]),
            aux_lr=float(config["optimizer"]["aux_lr"]),
            aux_betas=tuple(config["optimizer"]["aux_betas"]),
            aux_eps=float(config["optimizer"]["aux_eps"]),
            aux_weight_decay=float(config["optimizer"]["aux_weight_decay"]),
        ),
    )

    best_state = clone_state_dict(model)
    best_val_loss = float("inf")
    previous_subset_embeddings = None
    best_epoch = 0

    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_total, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, beta)
        val_total, val_recon, val_kl = evaluate_epoch(model, val_loader, device, beta)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = clone_state_dict(model)
            best_epoch = epoch

        generalization_loss = 0.0 if best_val_loss == 0 else 100.0 * ((val_total / best_val_loss) - 1.0)

        change_pct = None
        if str(config["early_stopping"]["method"]) == "embedding_stability":
            previous_subset_embeddings, change_pct = compute_embedding_change(
                model,
                subset_loader,
                device,
                embedding_batch,
                previous_subset_embeddings,
            )

        log_line = (
            f"epoch={epoch} "
            f"train_total={train_total:.6f} train_recon={train_recon:.6f} train_kl={train_kl:.6f} "
            f"val_total={val_total:.6f} val_recon={val_recon:.6f} val_kl={val_kl:.6f} "
            f"gl={generalization_loss:.3f}"
        )
        if change_pct is not None:
            log_line += f" embedding_change_pct={change_pct:.3f}"
        log(log_line)

        if epoch < int(config["min_epochs"]):
            continue

        method = str(config["early_stopping"]["method"])
        if method == "generalization_loss":
            stop = generalization_loss > float(config["early_stopping"]["generalization_loss_threshold"])
        elif method == "embedding_stability":
            stop = change_pct is not None and change_pct < float(config["early_stopping"]["cosine_change_threshold"])
        else:
            raise ValueError(f"unsupported early stopping method: {method}")

        if stop:
            log(f"early_stop epoch={epoch} method={method}")
            break

    model.load_state_dict(best_state)

    output_root = Path(__file__).resolve().parent / "data"
    output_paths: dict[str, Path] = {}

    for split_name, dataset in zip(DEFAULT_SPLITS, (train_dataset, val_dataset, test_dataset)):
        loader = build_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)
        output_path = output_root / f"variational_{data_dir.name}_{split_name}.parquet"
        output_paths[split_name] = export_embeddings_to_parquet(
            model,
            dataset,
            loader,
            device,
            output_path,
            infer_batch,
        )
        log(f"export split={split_name} path={output_path}")

    report(
        f"DONE module=variational best_epoch={best_epoch} best_val_total={best_val_loss:.6f} "
        f"training={output_paths['training']} validation={output_paths['validation']} test={output_paths['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
