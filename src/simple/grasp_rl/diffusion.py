"""Unconditional SMP diffusion prior and fixed-timestep guidance.

The denoiser and scheduler are adapted from SMP commit
``0e67286fe7df77a73740d237ef36b109136552b6``.  The public denoiser interface
remains strictly ``model(x_t, t)``; task/object conditions never enter it.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class _Timesteps(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=t.device) / half
        values = t.float()[:, None] * torch.exp(exponent)[None]
        return torch.cat([torch.cos(values), torch.sin(values)], dim=-1)


class _AdaLayerNormSingle(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time = _Timesteps(256)
        self.embed = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(self.time(t))).unsqueeze(1)


class _Position(nn.Module):
    def __init__(self, dim: int, length: int):
        super().__init__()
        position = torch.arange(max(length, 32)).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(1, max(length, 32), dim)
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]]


class _SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input = nn.Linear(dim, dim * 8)
        self.output = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(a) * b)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ff = _SwiGLU(dim)
        self.scale_shift = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        q = self.q(x).reshape(shape).transpose(1, 2)
        k = self.k(x).reshape(shape).transpose(1, 2)
        v = self.v(x).reshape(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v)
        return self.dropout(self.attn_out(output.transpose(1, 2).reshape(batch, tokens, dim)))

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        values = (self.scale_shift + modulation.reshape(batch, 1, 6, -1)).chunk(6, dim=-2)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = [v.squeeze(-2) for v in values]
        hidden = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self._attention(hidden)
        hidden = self.norm2(x) * (1 + scale_f) + shift_f
        return x + gate_f * self.ff(hidden)


class DiffusionDenoiser(nn.Module):
    """DiT epsilon predictor: ``(B,W,F),(B,) -> (B,W,F)``."""

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.pre = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
        self.input = nn.Linear(feature_dim, d_model, bias=False)
        self.time = _AdaLayerNormSingle(d_model)
        self.position = _Position(d_model, window_size)
        self.blocks = nn.ModuleList([_Block(d_model, nhead, dropout) for _ in range(num_layers)])
        self.output = nn.Linear(d_model, feature_dim, bias=False)
        self.post = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hidden = x_t.transpose(1, 2)
        hidden = (self.pre(hidden) + hidden).transpose(1, 2)
        hidden = self.position(self.input(hidden))
        modulation = self.time(t)
        for block in self.blocks:
            hidden = block(hidden, modulation)
        hidden = self.output(hidden).transpose(1, 2)
        return (self.post(hidden) + hidden).transpose(1, 2)


def _cosine_betas(steps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(value: float) -> float:
        return math.cos((value + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor(
        [min(1 - alpha_bar((i + 1) / steps) / alpha_bar(i / steps), max_beta) for i in range(steps)],
        dtype=torch.float32,
    )


class DDPMScheduler(nn.Module):
    def __init__(self, num_timesteps: int = 50):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = _cosine_betas(num_timesteps)
        alpha = 1 - betas
        cumulative = torch.cumprod(alpha, dim=0)
        self.register_buffer("sqrt_alpha", torch.sqrt(cumulative))
        self.register_buffer("sqrt_one_minus_alpha", torch.sqrt(1 - cumulative))

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shape = (-1, *([1] * (x_0.ndim - 1)))
        return self.sqrt_alpha[t].view(shape) * x_0 + self.sqrt_one_minus_alpha[t].view(shape) * noise


class MotionWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, processed_dir: str | Path, split: str):
        root = Path(processed_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        chunks = []
        for episode in manifest["splits"][split]:
            with np.load(root / "episodes" / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
                chunks.append(data["windows"].astype(np.float32))
        values = np.concatenate(chunks)
        with np.load(root / "norm_stats.npz", allow_pickle=False) as stats:
            self.q_low = stats["q_low"].astype(np.float32)
            self.q_high = stats["q_high"].astype(np.float32)
        values = 2 * (values - self.q_low) / (self.q_high - self.q_low + 1e-8) - 1
        self.windows = torch.from_numpy(values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


@dataclass
class DiffusionTrainConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    num_timesteps: int = 50
    num_noise_samples: int = 10
    batch_size: int = 1024
    max_epochs: int = 5000
    min_epochs: int = 1000
    patience: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda:0"
    resume: str | None = None


def _loss(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    batch = x_0.shape[0]
    expanded = x_0[:, None].expand(batch, samples, *x_0.shape[1:]).reshape(
        batch * samples, *x_0.shape[1:]
    )
    t = torch.randint(0, scheduler.num_timesteps, (len(expanded),), device=x_0.device)
    noise = torch.randn_like(expanded)
    return F.l1_loss(model(scheduler.add_noise(expanded, noise, t), t), noise)


@torch.no_grad()
def _validation_loss(model, scheduler, loader, device, samples: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        total += float(_loss(model, scheduler, batch.to(device), samples))
        count += 1
    return total / max(count, 1)


def train_diffusion(
    processed_dir: str | Path,
    output_dir: str | Path,
    config: DiffusionTrainConfig | None = None,
) -> Path:
    config = config or DiffusionTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    train_data = MotionWindowDataset(processed_dir, "train")
    val_data = MotionWindowDataset(processed_dir, "val")
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    feature_dim, window = train_data.windows.shape[-1], train_data.windows.shape[-2]
    model = DiffusionDenoiser(
        feature_dim, window, config.d_model, config.nhead, config.num_layers, config.dropout
    ).to(device)
    scheduler = DDPMScheduler(config.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    latest_path = output / "latest.pt"
    best = float("inf")
    best_epoch = -1
    history = []
    start_epoch = 0
    if config.resume:
        checkpoint: dict[str, Any] = torch.load(
            config.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best)))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        history_path = output / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
        if not best_path.exists():
            torch.save(
                {
                    "model": copy.deepcopy(model.state_dict()),
                    "q_low": train_data.q_low,
                    "q_high": train_data.q_high,
                    "cfg": {
                        **asdict(config),
                        "feature_dim": feature_dim,
                        "window_size": window,
                    },
                    "epoch": best_epoch,
                    "val_loss": best,
                    "best_val_loss": best,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        print(json.dumps({"resumed_from": config.resume, "start_epoch": start_epoch}))
    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            loss = _loss(model, scheduler, batch.to(device, non_blocking=True), config.num_noise_samples)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if epoch % 10 == 0 or epoch == config.max_epochs - 1:
            val = _validation_loss(model, scheduler, val_loader, device, config.num_noise_samples)
            row = {"epoch": epoch, "train": total / max(count, 1), "val": val}
            history.append(row)
            print(json.dumps(row))
            if val < best:
                best, best_epoch = val, epoch
                torch.save(
                    {
                        "model": copy.deepcopy(model.state_dict()),
                        "q_low": train_data.q_low,
                        "q_high": train_data.q_high,
                        "cfg": {**asdict(config), "feature_dim": feature_dim, "window_size": window},
                        "epoch": epoch,
                        "val_loss": val,
                        "best_val_loss": best,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "q_low": train_data.q_low,
                    "q_high": train_data.q_high,
                    "cfg": {**asdict(config), "feature_dim": feature_dim, "window_size": window},
                    "epoch": epoch,
                    "val_loss": val,
                    "best_val_loss": best,
                    "best_epoch": best_epoch,
                },
                latest_path,
            )
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if epoch >= config.min_epochs and epoch - best_epoch >= config.patience:
                break
    (output / "history.json").write_text(json.dumps(history, indent=2))
    evaluate_diffusion(best_path, processed_dir, output / "score_report.json", device=str(device))
    return best_path


def load_prior(path: str | Path, device: torch.device | str):
    checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    model = DiffusionDenoiser(
        cfg["feature_dim"],
        cfg["window_size"],
        cfg["d_model"],
        cfg["nhead"],
        cfg["num_layers"],
        cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    scheduler = DDPMScheduler(cfg["num_timesteps"]).to(device)
    q_low = torch.as_tensor(checkpoint["q_low"], device=device)
    q_high = torch.as_tensor(checkpoint["q_high"], device=device)
    return model, scheduler, q_low, q_high


class DiffNormalizer:
    def __init__(self, steps: int, device: torch.device | str):
        self.mean = torch.ones(steps, device=device)
        self.count = torch.zeros(steps, device=device, dtype=torch.long)

    def normalize(self, timestep: int, values: torch.Tensor) -> torch.Tensor:
        count = int(self.count[timestep])
        batch = values.numel()
        mean = values.mean()
        self.mean[timestep] = mean if count == 0 else (count * self.mean[timestep] + batch * mean) / (count + batch)
        self.count[timestep] += batch
        return values / self.mean[timestep].clamp_min(1e-4)


class Guidance:
    def __init__(self, checkpoint: str | Path, device: torch.device | str, ws: float = 4.0):
        self.model, self.scheduler, self.q_low, self.q_high = load_prior(checkpoint, device)
        self.normalizer = DiffNormalizer(self.scheduler.num_timesteps, device)
        self.ws = ws
        self.timesteps = (8, 15, 22)

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_0 = 2 * (features - self.q_low) / (self.q_high - self.q_low + 1e-8) - 1
        total, raw = torch.zeros(len(x_0), device=x_0.device), torch.zeros(len(x_0), device=x_0.device)
        for scalar in self.timesteps:
            t = torch.full((len(x_0),), scalar, device=x_0.device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            error = ((self.model(self.scheduler.add_noise(x_0, noise, t), t) - noise) ** 2).mean((-1, -2))
            raw += error
            total += self.normalizer.normalize(scalar, error)
        raw /= len(self.timesteps)
        return torch.exp(-self.ws * total / len(self.timesteps)), raw


@torch.no_grad()
def evaluate_diffusion(
    checkpoint: str | Path,
    processed_dir: str | Path,
    output_file: str | Path,
    device: str = "cuda:0",
) -> dict[str, float]:
    model, scheduler, _, _ = load_prior(checkpoint, device)
    dataset = MotionWindowDataset(processed_dir, "test")
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    real_errors, shuffled_errors = [], []
    for batch in loader:
        batch = batch.to(device)
        shuffled = batch[:, torch.randperm(batch.shape[1], device=batch.device)]
        for values, target in ((batch, real_errors), (shuffled, shuffled_errors)):
            errors = torch.zeros(len(values), device=values.device)
            for scalar in (8, 15, 22):
                t = torch.full((len(values),), scalar, device=values.device, dtype=torch.long)
                noise = torch.randn_like(values)
                errors += ((model(scheduler.add_noise(values, noise, t), t) - noise) ** 2).mean((-1, -2))
            target.extend((errors / 3).cpu().tolist())
    real = float(np.median(real_errors))
    shuffled = float(np.median(shuffled_errors))
    report = {"real_median_mse": real, "time_shuffle_median_mse": shuffled, "ratio": shuffled / max(real, 1e-8)}
    Path(output_file).write_text(json.dumps(report, indent=2))
    return report
