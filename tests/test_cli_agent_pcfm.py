from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app
from constelx.physics.booz_proxy import compute_proxies


def test_agent_with_pcfm_norm_constraint_runs_and_applies(tmp_path: Path) -> None:
    # Constrain (r_cos[1][5], z_sin[1][5]) to lie on circle of radius 0.06
    constraints = [
        {
            "type": "norm_eq",
            "radius": 0.06,
            "terms": [
                {"field": "r_cos", "i": 1, "j": 5, "w": 1.0},
                {"field": "z_sin", "i": 1, "j": 5, "w": 1.0},
            ],
        }
    ]
    constraints_path = tmp_path / "constraints.json"
    constraints_path.write_text(json.dumps(constraints))

    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "4",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--correction",
            "pcfm",
            "--constraints-file",
            str(constraints_path),
            "--pcfm-gn-iters",
            "12",
        ],
    )
    assert result.exit_code == 0

    # Verify proposals exist and first boundary satisfies the norm constraint approximately
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    out = subdirs[0]
    proposals = (out / "proposals.jsonl").read_text().splitlines()
    assert len(proposals) >= 1
    first = json.loads(proposals[0])
    b = first["boundary"]
    x = float(b["r_cos"][1][5])
    y = float(b["z_sin"][1][5])
    assert abs((x * x + y * y) - 0.06 * 0.06) < 1e-4


def test_agent_with_pcfm_ar_band_constraint(tmp_path: Path) -> None:
    constraints = [
        {
            "type": "ar_band",
            "major": {"field": "r_cos", "i": 0, "j": 4},
            "minor": [
                {"field": "r_cos", "i": 1, "j": 5},
                {"field": "z_sin", "i": 1, "j": 5},
            ],
            "amin": 4.0,
            "amax": 8.0,
        }
    ]
    constraints_path = tmp_path / "ar_band.json"
    constraints_path.write_text(json.dumps(constraints))

    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "4",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--correction",
            "pcfm",
            "--constraints-file",
            str(constraints_path),
            "--pcfm-gn-iters",
            "12",
        ],
    )
    assert result.exit_code == 0

    proposal_files = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert proposal_files
    proposals = (proposal_files[0] / "proposals.jsonl").read_text().splitlines()
    first = json.loads(proposals[0])
    b = first["boundary"]
    r0 = abs(float(b["r_cos"][0][4]))
    rc = float(b["r_cos"][1][5])
    zs = float(b["z_sin"][1][5])
    helical = (rc * rc + zs * zs) ** 0.5
    aspect = r0 / max(helical, 1e-8)
    assert 4.0 - 1e-2 <= aspect <= 8.0 + 1e-2


def test_agent_with_pcfm_proxy_band(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "4",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--correction",
            "pcfm",
            "--constraints-file",
            "examples/pcfm_qs_band.json",
            "--pcfm-gn-iters",
            "12",
        ],
    )
    assert result.exit_code == 0
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert run_dirs
    proposals = (run_dirs[0] / "proposals.jsonl").read_text().splitlines()
    assert proposals
    boundary = json.loads(proposals[0])["boundary"]
    proxies = compute_proxies(boundary)
    assert proxies.qs_residual <= 0.21


def test_agent_cli_rejects_unknown_mf_proxy_metric(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "2",
            "--runs-dir",
            str(runs_dir),
            "--mf-proxy",
            "--mf-proxy-metric",
            "invalid_metric",
        ],
    )
    assert result.exit_code != 0
    assert "must be one of" in result.output.lower()
