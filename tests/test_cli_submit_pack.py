from __future__ import annotations

import json
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_submit_pack_creates_zip_with_boundary(tmp_path: Path) -> None:
    # Run a tiny agent run to get artifacts
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    res = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "2",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert res.exit_code == 0
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    out = subdirs[0]

    # Pack submission
    sub_zip = tmp_path / "sub.zip"
    res2 = runner.invoke(app, ["submit", "pack", str(out), "--out", str(sub_zip)])
    assert res2.exit_code == 0
    assert sub_zip.exists()
    with zipfile.ZipFile(sub_zip, "r") as zf:
        names = set(zf.namelist())
        assert "boundary.json" in names
        assert "metadata.json" in names
        # boundary should be JSON-decodable
        b = json.loads(zf.read("boundary.json").decode("utf-8"))
        assert isinstance(b, dict) and "r_cos" in b and "z_sin" in b


def test_submit_pack_topk_writes_boundaries_jsonl(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    # Create a run with a few evaluations
    res = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "3",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert res.exit_code == 0
    out = next(p for p in runs_dir.iterdir() if p.is_dir())
    sub_zip = tmp_path / "sub_topk.zip"
    res2 = runner.invoke(app, ["submit", "pack", str(out), "--out", str(sub_zip), "--top-k", "2"])
    assert res2.exit_code == 0
    with zipfile.ZipFile(sub_zip, "r") as zf:
        names = set(zf.namelist())
        assert "boundary.json" in names
        assert "boundaries.jsonl" in names
        txt = zf.read("boundaries.jsonl").decode("utf-8").strip()
        lines = [json.loads(line) for line in txt.splitlines() if line.strip()]
        assert len(lines) >= 1
        assert all(isinstance(rec.get("boundary"), dict) for rec in lines)
