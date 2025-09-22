# PR: Flip defaults to real, add dev-only guards, and CI real-smoke

## Summary
- Flip CLI defaults to use the real evaluator and HF data by default.
- Add central dev-only guards and real-only enforcement (opt-in).
- Guard synthetic entrypoints (example boundary, CMA-ES sphere).
- Add packaging provenance guard with `--allow-dev`.
- Add real-smoke CI workflow (PR triggers + cache).
- Update README quickstart to real-first; add “Dev only” section.
- Robustness fixes: integer `nfp` in CSV and surrogate dtype coercion.

## Key Changes
- Defaults flipped to real
  - Data fetch default HF: `src/constelx/cli.py`
  - Eval forward defaults to real: `src/constelx/cli.py`
  - Agent/opt/ablate tri‑state (unspecified → True): `src/constelx/cli.py`
- Dev-only guards
  - Central helpers: `src/constelx/dev.py`
  - `--example` synthetic guarded: `src/constelx/cli.py`
  - CMA‑ES synthetic fixture guarded: `src/constelx/cli.py`
  - Packaging guard (`--allow-dev`): `src/constelx/cli.py`, `src/constelx/submit/pack.py`
- Pre-flight enforcement
  - `eval.forward` early check
  - `eval.forward_many` early checks (mf_proxy and explicit placeholder)
- CI real-smoke
  - `.github/workflows/real-smoke.yml` runs on PRs touching eval/physics with pip + evaluator cache
- Docs
  - README: real-first quickstart and Dev-only section
- Robustness
  - CSV `nfp` integer stability
  - Surrogate training dtype coercion

## Follow-ups
- Update `AGENTS.md` to mirror README real-first quickstart and guards.
- Split examples: move synthetic under `examples/dev/`, add `examples/real_quickstart/`.
- Gate real-smoke by label (optional) or ensure physics extras are available in CI.

