from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def _boundary_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if c.startswith("boundary.r_cos") or c.startswith("boundary.z_sin")
    ]


class MLP(nn.Module):  # type: ignore[misc]
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


def train_simple_mlp(cache_dir: Path, output_dir: Path) -> None:
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

    for _ in range(200):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(output_dir) / "mlp.pt")
