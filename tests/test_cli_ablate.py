from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_cli_ablate_quick_components(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "ablations"
    # Include a correction toggle to exercise constraint wiring
    result = runner.invoke(
        app,
        [
            "ablate",
            "run",
            "--nfp",
            "3",
            "--budget",
            "3",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--components",
            "guard_simple,mf_proxy,correction=eci_linear",
        ],
    )
    assert result.exit_code == 0

    # Find the timestamped ablation folder and summary
    roots = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(roots) == 1
    root = roots[0]
    summary_csv = root / "summary.csv"
    assert summary_csv.exists()
    content = summary_csv.read_text()
    # minimal presence checks
    assert "baseline" in content
    assert "guard_simple" in content
    assert "mf_proxy" in content
    assert "correction=eci_linear" in content or "eci_linear" in content

