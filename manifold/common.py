#!/usr/bin/env python3
#
# common.py  Andrew Belles  April 10th, 2026
#
# Shared modules for representation learning.
#

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from muon import adam_update, muon_update


MUON_MODULE_TYPES = (nn.Linear, nn.Conv2d)


def _resolve_activation(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }

    try:
        return activations[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"unsupported activation '{name}', expected one of: {supported}") from exc


def _coerce_positive_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        raise ValueError(f"{name} must contain at least one width")
    if any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain only positive values")
    return resolved


def _validate_input_shape(input_shape: Sequence[int]) -> tuple[int, int, int]:
    resolved = tuple(int(dim) for dim in input_shape)
    if len(resolved) != 3:
        raise ValueError(f"expected input_shape [channels, height, width], got {resolved}")
    channels, height, width = resolved
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"input_shape must be strictly positive, got {resolved}")
    return resolved  # type: ignore[return-value]


def _make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _repeat_static_input(x: torch.Tensor, num_steps: int) -> torch.Tensor:
    if x.ndim == 2 or x.ndim == 4:
        repeats = (num_steps,) + (1,) * x.ndim
        return x.unsqueeze(0).repeat(*repeats)

    if x.ndim == 3 or x.ndim == 5:
        if x.size(0) != num_steps:
            raise ValueError(f"expected time-major input with {num_steps} steps, got shape {tuple(x.shape)}")
        return x

    raise ValueError(
        "expected input shape [batch, features], [steps, batch, features], "
        "[batch, channels, height, width], or [steps, batch, channels, height, width], "
        f"got {tuple(x.shape)}"
    )


def compute_sparse_penalty(
    spikes: torch.Tensor,
    target_sparsity: float = 0.05,
    penalty: str = "kl",
) -> torch.Tensor:
    if spikes.ndim < 2:
        raise ValueError(f"expected spike tensor with at least 2 dims, got {tuple(spikes.shape)}")

    reduce_dims = (0,) if spikes.ndim == 2 else (0, 1)
    firing_rate = spikes.float().mean(dim=reduce_dims)

    if penalty == "l1":
        return firing_rate.abs().mean()

    if penalty == "kl":
        target = torch.full_like(firing_rate, target_sparsity).clamp(1e-6, 1 - 1e-6)
        firing_rate = firing_rate.clamp(1e-6, 1 - 1e-6)
        kl = target * torch.log(target / firing_rate)
        kl = kl + (1 - target) * torch.log((1 - target) / (1 - firing_rate))
        return kl.mean()

    raise ValueError(f"unsupported sparse penalty '{penalty}', expected 'kl' or 'l1'")


def standard_autoencoder_objective(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def spiking_autoencoder_objective(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    sparse_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(x_hat, x)
    total_loss = reconstruction_loss + sparse_loss
    return total_loss, reconstruction_loss, sparse_loss


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()

        activation_cls = _resolve_activation(activation)
        self.norm1 = _make_group_norm(channels)
        self.norm2 = _make_group_norm(channels)
        self.activation = activation_cls()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(x))
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.activation(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class EncoderDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu"):
        super().__init__()

        activation_cls = _resolve_activation(activation)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = _make_group_norm(out_channels)
        self.activation = activation_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class DecoderUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu"):
        super().__init__()

        activation_cls = _resolve_activation(activation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = _make_group_norm(out_channels)
        self.activation = activation_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


def _build_residual_stage(
    channels: int,
    block_depth: int,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    if block_depth <= 0:
        raise ValueError("block_depth must be positive")

    return nn.Sequential(
        *(ResidualConvBlock(channels, activation=activation, dropout=dropout) for _ in range(block_depth))
    )


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        latent_dim: int,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] | None = None,
        block_depth: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.input_shape = _validate_input_shape(input_shape)
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = _coerce_positive_sequence(channel_multipliers or (1, 2, 4, 8), "channel_multipliers")
        self.block_depth = int(block_depth)
        self.activation = activation
        self.dropout = dropout

        input_channels, _, _ = self.input_shape
        stage_channels = tuple(base_channels * multiplier for multiplier in self.channel_multipliers)
        self.stage_channels = stage_channels

        activation_cls = _resolve_activation(activation)
        self.stem = nn.Conv2d(input_channels, stage_channels[0], kernel_size=3, padding=1, bias=False)
        self.stem_norm = _make_group_norm(stage_channels[0])
        self.stem_activation = activation_cls()

        self.encoder_stages = nn.ModuleList(
            _build_residual_stage(channels, self.block_depth, activation, dropout) for channels in stage_channels
        )
        self.encoder_downsamples = nn.ModuleList(
            EncoderDownsample(stage_channels[index], stage_channels[index + 1], activation=activation)
            for index in range(len(stage_channels) - 1)
        )

        with torch.no_grad():
            encoded = self._encode_features(torch.zeros(1, *self.input_shape))
        self.encoded_shape = tuple(int(dim) for dim in encoded.shape[1:])
        self.encoded_dim = math.prod(self.encoded_shape)

        self.to_latent = nn.Linear(self.encoded_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.encoded_dim)

        reversed_channels = tuple(reversed(stage_channels))
        self.decoder_stages = nn.ModuleList(
            _build_residual_stage(channels, self.block_depth, activation, dropout) for channels in reversed_channels
        )
        self.decoder_upsamples = nn.ModuleList(
            DecoderUpsample(reversed_channels[index], reversed_channels[index + 1], activation=activation)
            for index in range(len(reversed_channels) - 1)
        )

        self.final_norm = _make_group_norm(reversed_channels[-1])
        self.final_activation = activation_cls()
        self.final_conv = nn.Conv2d(reversed_channels[-1], input_channels, kernel_size=3, padding=1)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_norm(x)
        x = self.stem_activation(x)

        for index, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if index < len(self.encoder_downsamples):
                x = self.encoder_downsamples[index](x)

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode_features(x)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        x = self.from_latent(z).view(batch_size, *self.encoded_shape)

        for index, stage in enumerate(self.decoder_stages):
            x = stage(x)
            if index < len(self.decoder_upsamples):
                x = self.decoder_upsamples[index](x)

        x = F.interpolate(x, size=self.input_shape[1:], mode="bilinear", align_corners=False)
        x = self.final_norm(x)
        x = self.final_activation(x)
        return self.final_conv(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class SpikingAutoEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        latent_dim: int,
        encoder_channels: Sequence[int] | None = None,
        num_steps: int = 8,
        beta: float = 0.95,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        sparse_lambda: float = 1e-3,
        target_sparsity: float = 0.05,
        sparse_penalty: str = "kl",
    ):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if sparse_lambda < 0:
            raise ValueError("sparse_lambda must be non-negative")
        if not 0 < target_sparsity < 1:
            raise ValueError("target_sparsity must be in (0, 1)")

        self.input_shape = _validate_input_shape(input_shape)
        self.latent_dim = latent_dim
        self.encoder_channels = _coerce_positive_sequence(encoder_channels or (32, 64), "encoder_channels")
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.sparse_lambda = sparse_lambda
        self.target_sparsity = target_sparsity
        self.sparse_penalty_name = sparse_penalty

        input_channels, _, _ = self.input_shape
        encoder_dims = (input_channels, *self.encoder_channels, latent_dim)
        self.encoder_convs = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for in_channels, out_channels in zip(encoder_dims[:-1], encoder_dims[1:])
        )
        self.encoder_neurons = nn.ModuleList(
            snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
            for _ in self.encoder_convs
        )
        self.encoder_pools = nn.ModuleList(nn.AvgPool2d(kernel_size=2, stride=2) for _ in self.encoder_convs)

        decoder_dims = (latent_dim, *reversed(self.encoder_channels), input_channels)
        self.decoder_convs = nn.ModuleList(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            for in_channels, out_channels in zip(decoder_dims[:-1], decoder_dims[1:])
        )
        self.decoder_neurons = nn.ModuleList(
            snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
            for _ in self.decoder_convs[:-1]
        )

    def sparse_loss(self, latent_spikes: torch.Tensor) -> torch.Tensor:
        return self.sparse_lambda * compute_sparse_penalty(
            latent_spikes,
            target_sparsity=self.target_sparsity,
            penalty=self.sparse_penalty_name,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        snn.Leaky.reset_hidden()
        snn.Leaky.detach_hidden()
        inputs = _repeat_static_input(x, self.num_steps)
        encoder_mems: list[torch.Tensor | None] = [None] * len(self.encoder_neurons)
        latent_spikes: list[torch.Tensor] = []

        for step in range(self.num_steps):
            cur = inputs[step]
            for index, (conv, neuron, pool) in enumerate(zip(self.encoder_convs, self.encoder_neurons, self.encoder_pools)):
                cur = conv(cur)
                spk, mem = neuron(cur, encoder_mems[index])
                encoder_mems[index] = mem
                cur = pool(spk)
            latent_spikes.append(cur)

        spike_tensor = torch.stack(latent_spikes, dim=0)
        latent_rate_map = spike_tensor.float().mean(dim=0)
        latent_rate = F.adaptive_avg_pool2d(latent_rate_map, output_size=1).flatten(start_dim=1)
        return latent_rate, spike_tensor

    def decode(self, latent_spikes: torch.Tensor) -> torch.Tensor:
        if latent_spikes.ndim != 5:
            raise ValueError(
                "expected latent spikes with shape [steps, batch, channels, height, width], "
                f"got {tuple(latent_spikes.shape)}"
            )

        decoder_mems: list[torch.Tensor | None] = [None] * len(self.decoder_neurons)
        recon_steps: list[torch.Tensor] = []

        for step in range(latent_spikes.size(0)):
            cur = latent_spikes[step]
            for index, conv in enumerate(self.decoder_convs):
                cur = conv(cur)
                if index < len(self.decoder_neurons):
                    spk, mem = self.decoder_neurons[index](cur, decoder_mems[index])
                    decoder_mems[index] = mem
                    cur = spk

            if cur.shape[-2:] != self.input_shape[1:]:
                cur = F.interpolate(cur, size=self.input_shape[1:], mode="bilinear", align_corners=False)
            recon_steps.append(cur)

        return torch.stack(recon_steps, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_rate, latent_spikes = self.encode(x)
        x_hat = self.decode(latent_spikes)
        sparse_loss = self.sparse_loss(latent_spikes)
        return x_hat, latent_rate, sparse_loss


@dataclass(frozen=True)
class MuonConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    aux_lr: float = 3e-4
    aux_betas: tuple[float, float] = (0.9, 0.95)
    aux_eps: float = 1e-10
    aux_weight_decay: float = 0.0


@dataclass(frozen=True)
class MuonParameterGroups:
    muon_names: tuple[str, ...]
    aux_names: tuple[str, ...]


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict[str, object]]):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    if p.ndim == 4 and update.ndim == 2:
                        update = update.view_as(p)

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])


def _iter_muon_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for module_name, module in model.named_modules():
        if not isinstance(module, MUON_MODULE_TYPES):
            continue

        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter) or not weight.requires_grad:
            continue

        name = f"{module_name}.weight" if module_name else "weight"
        yield name, weight


def partition_named_muon_parameters(model: nn.Module) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    muon_named_params = list(_iter_muon_named_parameters(model))
    muon_param_ids = {id(param) for _, param in muon_named_params}

    aux_named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in muon_param_ids
    ]

    return muon_named_params, aux_named_params


def describe_muon_parameter_groups(model: nn.Module) -> MuonParameterGroups:
    muon_named_params, aux_named_params = partition_named_muon_parameters(model)
    return MuonParameterGroups(
        muon_names=tuple(name for name, _ in muon_named_params),
        aux_names=tuple(name for name, _ in aux_named_params),
    )


def build_single_device_muon_optimizer(
    model: nn.Module,
    config: MuonConfig | None = None,
) -> torch.optim.Optimizer:
    resolved = config or MuonConfig()
    muon_named_params, aux_named_params = partition_named_muon_parameters(model)

    if not muon_named_params and not aux_named_params:
        raise ValueError("model has no trainable parameters")

    param_groups: list[dict[str, object]] = []

    if muon_named_params:
        param_groups.append(
            {
                "params": [param for _, param in muon_named_params],
                "lr": resolved.muon_lr,
                "momentum": resolved.muon_momentum,
                "weight_decay": resolved.muon_weight_decay,
                "use_muon": True,
            }
        )

    if aux_named_params:
        param_groups.append(
            {
                "params": [param for _, param in aux_named_params],
                "lr": resolved.aux_lr,
                "betas": resolved.aux_betas,
                "eps": resolved.aux_eps,
                "weight_decay": resolved.aux_weight_decay,
                "use_muon": False,
            }
        )

    return SingleDeviceMuonWithAuxAdam(param_groups)
