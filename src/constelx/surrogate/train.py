from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ..physics.pbfm import conflict_free_update


def _boundary_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if c.startswith("boundary.r_cos") or c.startswith("boundary.z_sin")
    ]


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


def train_simple_mlp(
    cache_dir: Path, output_dir: Path, *, use_pbfm: bool = False, steps: int = 20
) -> None:
    df = pd.read_parquet(Path(cache_dir) / "subset.parquet")
    X = df[_boundary_cols(df)].fillna(0.0).to_numpy()
    # Toy target: pick one available scalar metric if present, else zeros
    target_col = next(
        (c for c in df.columns if c.startswith("metrics.") and df[c].dtype != object),
        None,
    )
    if target_col is None:
        y = torch.zeros(len(df), 1)
    else:
        y = torch.tensor(df[target_col].fillna(0.0).to_numpy()).float().view(-1, 1)

    X = torch.tensor(X).float()
    model = MLP(X.shape[1], 1)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    for _ in range(int(steps)):
        if not use_pbfm:
            opt.zero_grad()
            pred = model(X)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()
            continue

        # PBFM path: combine FM loss and a toy residual loss via conflict-free update
        # FM loss: match target
        opt.zero_grad()
        pred = model(X)
        fm_loss = ((pred - y) ** 2).mean()
        fm_loss.backward()
        g_fm = [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]

        # Residual loss: encourage small outputs (acts as a physics-like residual placeholder)
        opt.zero_grad()
        pred2 = model(X)
        resid_loss = (pred2**2).mean()
        resid_loss.backward()
        g_r = [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]

        # Replace gradients with conflict-free combination and step
        opt.zero_grad()
        for p, gf, gr in zip(model.parameters(), g_fm, g_r):
            # Compute normalized combination on CPU using numpy
            gf_np = gf.detach().cpu().numpy()
            gr_np = gr.detach().cpu().numpy()
            upd_np = conflict_free_update(gf_np, gr_np)
            upd = torch.from_numpy(upd_np).to(p)
            p.grad = upd
        opt.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(output_dir) / "mlp.pt")
