# Strategy — Agent and Research Roadmap

TL;DR (repo‑aligned)
- Train simple, fast surrogates on ConStellaration‑style data (MLP baseline available) and optionally add a PBFM‑style conflict‑free residual term during training.
- Enforce constraints at inference via correction hooks: ECI (linear Ax=b projection) and PCFM (nonlinear Gauss–Newton projection). Both are wired into `constelx agent run` with `--correction eci_linear|pcfm` and JSON specs (see `examples/pcfm_*.json`).
- Improve throughput with multi‑fidelity proxy gating (`--mf-proxy ...`) using Boozer proxy
  residuals (`qs_residual`, `qi_residual`, `helical_energy`, `mirror_ratio`) or placeholder
  metrics via the unified facade, novelty gating, and optional surrogate screening before
  evaluator calls. Use VMEC++ hot‑restart and a neighbor‑reuse cache to reduce per‑candidate
  cost.
- Optimize with currently implemented baselines (trust‑constr, ALM, CMA‑ES) and the ablation harness; prioritize feasibility‑first TR‑BO (FuRBO/BoTorch) next. For P3, track hypervolume explicitly via qEHVI or ε‑constraint once feasibility is reliable.
- Keep runs reproducible and packageable: the agent writes `runs/<ts>/...` and `constelx submit pack` creates the submission zip. Default runs are deterministic; optional LLM planners should interface by emitting ablation specs for `constelx ablate run`.

What’s implemented vs. planned
- Implemented now
  - Constraint hooks: `eci_linear` and `pcfm` under `constelx.agents.corrections.*`.
  - Multi‑fidelity gating, caching, and real‑evaluator timeouts/retries.
  - Surrogate baseline with optional PBFM conflict‑free update (`constelx surrogate train --use-pbfm`).
  - Optimizers: `opt cmaes` + `opt run --baseline trust-constr|alm`.
  - Boozer proxies: bounded QS/QI heuristics via `constelx.physics.booz_proxy` with optional real-evaluator hooks.
  - Agent loop: resume, geometry guards, novelty gating, surrogate screening, NFP round‑robin; artifacts and schema verified by tests.
  - Submission packaging: `constelx submit pack` (supports `--top-k`).
- Planned (next steps)
  - Feasibility‑first TR‑BO (FuRBO/BoTorch) and NSGA‑II/III harnesses.
  - qEHVI / ε‑constraint for P3 with explicit hypervolume tracking.
  - Surrogate portfolio: LightGBM/LightGBM‑LSS + SHAP for interpretability.
  - VMEC++ hot‑restart token + neighbor‑reuse cache in evaluator path.
  - Stage‑II coil optimization (SIMSOPT augmented Lagrangian) + coil‑simplicity surrogate.
  - Optional LLM‑assisted planner that emits ablation specs for `constelx ablate run`.

Practical CLI mapping
- ECI: `constelx agent run --correction eci_linear --constraints-file constraints.json`
- PCFM: `constelx agent run --correction pcfm --constraints-file examples/pcfm_norm.json [--pcfm-gn-iters ...]` (extras like `examples/pcfm_qs_band.json` gate the Boozer-derived QS residual directly).
- Multi‑fidelity: `constelx agent run --mf-proxy --mf-quantile 0.3`
- Surrogate training: `constelx surrogate train --out-dir outputs/surrogates/mlp --use-pbfm`
- Baselines: `constelx opt run --baseline trust-constr --use-physics --problem p1`
- Problems listing: `constelx eval problems` (shows P1–P3 expected metrics)
- Ablation harness: `constelx ablate run` (toggle components like guards, mf_proxy)

Challenge essentials & baselines (concise)
- P1 Geometric: minimize max elongation with constraints on aspect ratio A, average triangularity \bar{\delta}, and edge rotational transform per field period \tilde\iota/N_{\rm fp}.
- P2 Simple‑to‑build QI: minimize e_{L\nabla B} (coil‑simplicity proxy) under bounds on QI residual, mirror ratio \Delta, \tilde\iota/N_{\rm fp}, and elongation.
- P3 MHD‑stable QI (multi‑objective): improve (−e_{L\nabla B}, A) subject to vacuum well and a turbulence proxy; leaderboard uses hypervolume over feasible points.
- Baseline (paper): Augmented‑Lagrangian + NGOpt most reliably finds feasibility. We beat it by hard‑constrained generation (ECI/PCFM), multi‑fidelity screening, and feasibility‑first TR‑BO; VMEC++ remains the verifier.

Multi‑fidelity & hot‑restart (throughput)
- Use low/medium fidelities (reduced mode sets or relaxed tolerances) to gate expensive calls; reserve high fidelity for finalists. Enable with `--mf-proxy` and a quantile cutoff (e.g., `--mf-quantile 0.3`).
- Reuse VMEC++ restart state for nearby shapes; add a neighbor‑reuse cache keyed by a boundary fingerprint. Always log both proxy and “real” metrics with `phase=proxy|real`.

Runtime toggles (env)
- `CONSTELX_REAL_TIMEOUT_MS`, `CONSTELX_REAL_RETRIES`, `CONSTELX_REAL_BACKOFF` control real-evaluator timeout/retry/backoff.
- `CONSTELX_VMEC_VERBOSE=1` enables verbose VMEC++ logs; `CONSTELX_EVAL_LOG_DIR=/path/to/logs`
  writes one JSON per evaluation (boundary hash, metrics, VMEC knobs) for parity and
  regression debugging.
- Physics tests opt‑in: `CONSTELX_RUN_PHYSICS_TESTS=1`.

Playbooks (ready‑to‑run)
- P1: `constelx agent run --problem p1 --correction pcfm --constraints-file examples/pcfm_norm.json --mf-proxy --surrogate-screen --penalty-highm 1e-3`
- P2: `constelx agent run --problem p2 --correction pcfm --constraints-file examples/pcfm_ratio.json --mf-proxy --surrogate-screen --penalty-helical 5e-3`
- P3: `constelx agent run --problem p3 --correction pcfm --constraints-file examples/pcfm_qs_band.json --mf-proxy --surrogate-screen --top-k 8`

Packaging & provenance (recap)
- Artifacts: `runs/<ts>/config.yaml`, `proposals.jsonl`, `metrics.csv`, `best.json`, plus `README.md` with CLI/env info.
- CSV columns include `nfp,evaluator_score,agg_score,elapsed_ms,feasible,fail_reason,source`; with MF gating also log `phase=proxy|real`, `proxy_metric`, `proxy_score`. Surrogate‑filtered rows use `phase=surrogate`, include `surrogate_score`, and set `fail_reason=filtered_surrogate`.
- Submission: `constelx submit pack runs/<ts> --out submissions/<name>.zip --top-k K` writes `boundary.json`, `metadata.json {problem,scoring_version,git_sha,top_k}`, and `boundaries.jsonl` when `--top-k>1`.
- When novelty gating is enabled, `novelty.jsonl` persists novelty vectors for reuse across runs.

Risks & mitigations
- Boozer/VMEC metric drift: always VMEC++‑verify before archiving/submission; log proxy vs. real metrics.
- PCFM stability: cap GN steps/damping via CLI or JSON; start with ECI for linear bands; reject when KKT residual exceeds tolerance.
- Compute spikes: throttle with `--mf-quantile` and novelty gating; leverage restart tokens and neighbor cache.

This document is the higher-level strategy for ConStelX: it frames the
challenge, physics heuristics, model/agent portfolio, and a self-improving
agentic loop. It complements the engineering roadmap in `docs/ROADMAP.md`,
which tracks concrete milestones, CLI behaviors, module work, and tests that
land in this repository. When in doubt: use `docs/ROADMAP.md` to plan PRs and
CI work; use this strategy to guide research directions and longer-term
experiments.

—

Below I’ve done four things for you: 1. Mapped the ConStellaration Challenge: what it is, what’s provided, and how it’s scored. 2. Distilled the most useful ideas from the papers you uploaded—both the stellarator/physics ones and the two “agent” papers—and translated them into techniques directly applicable to ConStellaration. 3. Laid out concrete, end‑to‑end plans to top the leaderboard in each task (P1–P3), with careful, physics‑based reasoning for why these should work. 4. Designed a self‑improving “Auto‑Stellarator Engineer” (ASE) system that continuously generates better stellarators by closing the loop between surrogate modeling, VMEC++ evaluation, and LLM agents (leveraging the two agent papers).

⸻

0. What the ConStellaration Challenge is (and what you get)
   • The challenge (by Proxima Fusion + Hugging Face) invites participants to propose stellarator boundary shapes (and coils, where applicable) that optimize figures of merit (FOMs) computed by VMEC++, Proxima’s high‑performance equilibrium code. You submit shapes; an official evaluator computes metrics and ranks you on a live leaderboard. ￼
   • Proxima’s constellaration repo provides: a forward model (VMEC++ bindings), benchmark scoring functions, datasets (a large database of equilibria), examples, and ready‑to‑use Docker images for reproducibility. ￼
   • VMEC++ itself is the accelerated equilibrium solver you’ll be calling in the loop; Proxima’s repo emphasizes speed, numerical robustness, hot‑restart and HPC‑friendliness—all crucial for high‑throughput AI search. ￼

The three leaderboard problems (as defined in the dataset paper / docs):
• P1 – Geometry: minimize maximum cross‑sectional elongation under constraints on aspect ratio (A), edge rotational transform per field period (\tilde\iota), and average triangularity (\bar\delta). This isolates boundary shaping separate from plasma physics. ￼
• P2 – QI with simple coils: minimize the normalized inverse gradient‑scale metric e*{L\nabla B} (a coil‑complexity proxy; smaller is “simpler coils”), while keeping the QI‑residual small and mirror ratio \Delta bounded; geometric smoothness constraints apply. The normalization uses a/N*{\mathrm{fp}}. ￼
• P3 – Multi‑objective (Pareto): co‑optimize compactness (low A), coil simplicity (low e*{L\nabla B}), and physics constraints (vacuum magnetic well W*{\mathrm{MHD}} and flux compression \langle \chi\_{\nabla r}\rangle). The aim is to trace a Pareto front (tradeoffs) at high fidelity. ￼

The database released with the challenge includes a large number of VMEC equilibria parameterized by boundary Fourier coefficients R*{m,n}, Z*{m,n} (a.k.a. RBC*{m,n}, ZBS*{m,n}), along with labeled physics metrics. This lets you train surrogates and generative models and then close the loop with VMEC++. ￼

⸻

1. What we should take from the papers (and how to use it)

1A. The stellarator & dataset papers
• ConStellaration dataset paper: gives the exact objective/constraint definitions for P1–P3, including the QI‑residual, mirror ratio \Delta, normalized e*{L\nabla B} (normalized by a/N*{\mathrm{fp}}), and the multi‑objective constraints involving W*{\mathrm{MHD}} and \langle \chi*{\nabla r}\rangle. Use these to build loss functions and constraint checkers for surrogates and acquisition criteria.
• The equilibria database paper: clearly sets the boundary parameterization (Fourier modes for R(\theta,\phi), Z(\theta,\phi)), the domain for coefficients, and describes correlations between metrics (e.g., quasi‑symmetry/QI trends vs mirror behavior). Use this to pick sensible priors for generative models (amplitude bounds, dominant modes), and to augment features (e.g., axis curvature/torsion proxies computed from low‑order modes). ￼
• Landreman–Charkiw “Stellarator optimization” lectures: provide the physics heuristics that connect shaping to neoclassical transport and coil complexity: relationships among QI / QS, mirror ratio, and \nabla B scale length; impact of aspect ratio and field period N*{\mathrm{fp}} on coil “smoothness.” These motivate why lowering e*{L\nabla B} (for fixed A, N\_{\mathrm{fp}}) tends to reduce winding complexity and ripple.

Key physics takeaway used later in the search design: in QP/QI machines, the characteristic \nabla B scale length tracks the geometric scale L*{\nabla B}\sim R_0/N*{\mathrm{fp}}. Thus the dimensionless e*{L\nabla B}\sim a/L*{\nabla B}\propto (a N*{\mathrm{fp}})/R_0. For fixed A\equiv R_0/a and N*{\mathrm{fp}}, the “global” scaling is set, so improvement must come from local smoothing of \nabla B via boundary shaping that respects QI/QS; this is exactly what P2 optimizes. ￼

1B. Two agent papers we can operationalize
• MLE‑STAR (Search + Targeted Refinement): a practical recipe for agentic code improvement. It (i) retrieves external knowledge to seed a good initial solution, (ii) runs nested refinement loops that ablate/modify one component at a time, and (iii) ensembles multiple solutions via an agent‑planned scheme. Also includes debugging and leakage checks—useful for safe automation. We will port these ideas to code blocks like “surrogate feature set,” “acquisition function,” “constraint projection,” “VMEC++ solver settings,” etc.
• ASI‑ARCH (autonomous research loop): a full closed‑loop science system with roles—Researcher (propose), Engineer (run and fix), Analyst (synthesize & update a cognition base), and an LLM‑as‑judge factor in the fitness. We’ll reuse the fitness blending idea (quantitative + qualitative) to guard against reward hacking (e.g., designs that “game” a surrogate but fail at high‑fidelity). The paper also demonstrates scaling laws for discovery under compute; that argues for parallel, multi‑fidelity search with curation and memory.

1C. Generative/PDE ideas that fit our search
• Physics‑Based Flow Matching (PBFM) (a.k.a. Flow Matching Meets PDEs): trains a velocity field to morph an easy base distribution into a target distribution constrained by physics/PDE structure. For our task, this is a principled way to sample boundary coefficients that already respect constraints (e.g., target A,\tilde\iota,\bar\delta in P1, or QI bounds in P2/P3) before any expensive evaluations. ￼
• Physics‑Constrained Flow Matching (PCFM): generalizes FM to Riemannian manifolds / constraint sets by projecting velocities onto the tangent space of constraints so the flow stays on‑manifold (constraint preservation via \frac{d}{dt}g(x)=\nabla g\cdot v=0). We’ll use PCFM to sample exactly feasible candidates under hard constraints (e.g., fixed A, \tilde\iota, \bar\delta; or hard QI residual caps) given differentiable surrogate constraints. ￼

⸻

2. A practical plan to top the leaderboard (P1–P3)

I’ll describe an incremental stack—start with safe & fast methods (good baselines), then plug in more sophisticated components to climb.

Common infrastructure (for all P1–P3)
• Pinned toolchain: use the official containers and VMEC++ bindings in the constellaration repo to match leaderboard evaluation exactly; exploit hot‑restart and consistent numerics. ￼
• Design space: parameterize by the same Fourier modes as the dataset (respect domain bounds for each R*{m,n}, Z*{m,n}). Include symmetry and N*{\mathrm{fp}} as fixed per‑task (or specified). Seed from database exemplars near the target constraints. ￼
• Fast multi‑head surrogates: one network predicting all task metrics: \max elongation, e*{L\nabla B}, QI residual, \Delta, W*{\mathrm{MHD}}, \langle \chi*{\nabla r}\rangle, \tilde\iota, \bar\delta, A. Use conformal prediction (quantile heads) to get uncertainty; add a Lipschitz penalty in coefficient space to improve local gradient fidelity (needed for PCFM projection). Train on the provided equilibria, then continually retrain with your own VMEC++ results (active learning). ￼
• Feature engineering (from physics): beyond raw coefficients, compute cheap axis proxies (from low‑order m=0,1 modes), approximate curvature/torsion, mirror indicators, near‑axis QS/QI proxies. This makes surrogates more sample efficient (per Landreman–Charkiw heuristics).
• Trust regions: keep candidates within training distribution measured by Mahalanobis distance in feature space. Promote exploration with a small, scheduled probability of leaving the trust region (but always validating with VMEC++).
• Multi‑fidelity evaluation: early screening with looser VMEC++ tolerances / coarser resolutions, promote to high‑fidelity only for top‑k. This is explicitly encouraged by the challenge repo and is a standard trick for search throughput. ￼

⸻

P1: Minimize maximum cross‑section elongation (constraints: A,\tilde\iota,\bar\delta)

Baseline
• Train a P1 surrogate for \max*{\phi}\kappa(\phi) (elongation) with inputs R*{m,n},Z\_{m,n}.
• Run a constrained CMA‑ES or trust‑region Bayesian optimization (BO) using surrogate‑predicted elongation + hard penalty for constraint violation (exact values verified by cheap VMEC++ pass).
• Use randomized cross‑section sampling in \phi to approximate the max; resample at each iteration (acts like “max‑pooling” during optimization).

Upgrade 1 — PCFM sampling on the constraint manifold
• Define constraint functions \(g(x)=[A(x)-A^\,\ \tilde\iota(x)-\tilde\iota^\,\ \bar\delta(x)-\bar\delta^\*]\). Using surrogate autodiff, project FM velocity v*\theta(x,t) onto the null space of \nabla g so that candidates stay feasible by construction. Then rank by predicted \max elongation; send the top N to VMEC++ for truth. (This preserves feasibility because \tfrac{d}{dt}g(x(t))=\nabla g\cdot v*{\perp}=0.) ￼

Upgrade 2 — Physics structure
• From Landreman–Charkiw, circular cross‑sections minimize elongation but may violate \tilde\iota,\bar\delta; use ellipticity‑modulated modes (m=2) only where needed to hit \tilde\iota,\bar\delta, keeping higher m small to avoid spikes in local elongation. Build this into a mode‑sparsity prior (L1 on high‑m) in the search.

Why this should win (theory)
• The KKT conditions for minimizing a \max-type objective under smooth constraints imply active‑set localization: the worst‑\phi cross‑section dominates the gradient. Penalizing high‑m amplitudes reduces local curvature bursts that create elongation spikes, while PCFM keeps (A,\tilde\iota,\bar\delta) exactly fixed. This concentrates the search where changes help the worst section, not the average.

⸻

P2: QI with simple coils (minimize e\_{L\nabla B} under QI‑residual + mirror constraints)

Baseline
• Multi‑objective surrogate for (e\_{L\nabla B},\ \text{QIres},\ \Delta).
• Constrained BO (e.g., augmented Lagrangian or q‑NEHVI with feasibility) over the boundary coefficients. Validate top picks with VMEC++.

Upgrade 1 — PCFM + PBFM hybrid generator
• Train a PCFM generator to sample feasible designs with QIres \le \epsilon and \Delta\le \Delta*{\max} (constraints defined in ConStellaration), then bias the flow toward lower e*{L\nabla B} using a temperature or reward‑weighted flow matching density. This yields high‑quality diverse proposals before any optimizer kicks in.
• Run short CMA‑ES from each sample as local polishers (5–15 steps), with a sparsity prior on high‑m harmonics to avoid local \nabla B spikes.

Upgrade 2 — Near‑axis/QI heuristics from physics
• For fixed A,N*{\mathrm{fp}}, local smoothing of \nabla B (hence lower e*{L\nabla B}) is achieved by limiting beat frequencies between harmonics n near N\_{\mathrm{fp}}, consistent with QI surfaces in Boozer coordinates; penalize specific phase misalignments that increase mirror ratio (learn these phase penalties from the dataset via SHAP/feature attribution).

Why this should win (theory)
• In QI/QS regimes, the bounce‑averaged drift vanishes to leading order, which correlates with reduced mirror ratio and smoother \nabla B. Since e*{L\nabla B}\propto a/L*{\nabla B} and L*{\nabla B}\sim R_0/N*{\mathrm{fp}}, for fixed (A,N\_{\mathrm{fp}}) the remaining handle is shape‑induced local variation. PCFM maintains the QI feasibility region while the objective concentrates mass on low‑variation B fields, so convergence pressure is correctly aligned with physics. ￼

⸻

P3: Pareto front (coil simplicity vs compactness, with W*{\mathrm{MHD}} and \langle \chi*{\nabla r}\rangle constraints)

Baseline
• Learn surrogates for e*{L\nabla B}, A, W*{\mathrm{MHD}}, \langle \chi\_{\nabla r}\rangle.
• Use q‑EHVI (or Tchebycheff scalarization with epsilon‑constraints) to trace the Pareto frontier; verify with VMEC++.

Upgrade 1 — Adaptive constraint handling
• Use feasibility classifiers (conformal) for (W*{\mathrm{MHD}}<0,\ \langle \chi*{\nabla r}\rangle\le c) to screen cheap. For borderline cases, upweight uncertainty exploration to learn the constraint boundaries sharply (this often yields the largest Pareto gains).

Upgrade 2 — Two‑stage exploration/verification (ASI‑ARCH style)
• Stage 1: explore widely at low fidelity, accumulate candidates and cluster the frontier in surrogate space to ensure diversity of tradeoffs.
• Stage 2: scale up fidelity and re‑rank by true hypervolume contribution. Maintain a memory (cognition base) of what kinds of shapes improve which metric without hurting constraints; feed this back into proposal prompts. ￼

Why this should win (theory)
• Pareto discovery benefits from diversity‑preserving acquisition; with accurate feasibility boundaries, the hypervolume gradient focuses evaluation budget near true frontier tangents. The physics constraints (well depth, flux compression) are smooth functionals of equilibrium fields, so multi‑fidelity surrogate learning converges quickly with active boundary sampling. ￼

⸻

3. The “Auto‑Stellarator Engineer” (ASE): a self‑improving system

ASE is a multi‑agent loop that continuously proposes, evaluates, learns, and improves—adapting MLE‑STAR and ASI‑ARCH to stellarator design:

Roles and memory (ASI‑ARCH‑inspired)
• Researcher: proposes design moves (edits to R*{m,n},Z*{m,n}, symmetry options, phase shifts, mode sparsification, PCFM/PBFM hyperparameters). Seeds moves from a cognition base distilled from the dataset and past runs (e.g., “to lower \Delta, phase‑align n=N*{\mathrm{fp}} modes across \phi”).
• Engineer: runs VMEC++; if it crashes or diverges, the self‑revision loop patches settings (mesh, tolerances), retries (leveraging hot‑restart), or simplifies modes until convergence—mirroring ASI‑ARCH’s “fix your own errors.” ￼
• Analyst: updates surrogates and uncertainty; writes short factual summaries of what worked (e.g., “reducing Z*{2,1} amplitude decreased e\_{L\nabla B} but raised \tilde\iota by 0.02”) to feed back into Researcher. ￼
• Judge (optional): an LLM “quality check” to flag candidates that game the surrogate yet look unphysical (e.g., excessive high‑m energy or near self‑intersection). This mimics ASI‑ARCH’s quantitative + qualitative fitness trick to reduce reward hacking. ￼

Targeted refinement loops (MLE‑STAR‑inspired)
• Treat the codebase as components: Featureizer, Surrogate, Acquisition, ConstraintProjection (PCFM), Generator (PBFM), LocalOptimizer, VMECSettings, Post‑filters.
• Run ablation‑guided refinement: each loop focuses on one component (e.g., “try alternative features: add near‑axis curvature proxy”), measures delta on a validation slate, keeps improvements. ￼
• Run ensembling across pipelines: combine top‑k proposal streams (e.g., CMA‑ES, PCFM‑only, BO) with LLM‑planned mixture (e.g., weights by recent hit rate, novelty, and surrogate uncertainty), just like MLE‑STAR’s ensemble planner. ￼

Constraint‑aware generative core
• PCFM sampler on manifold g(x)=0 (exact feasibility under surrogate constraints). PBFM shapes the flow density to favor lower task loss; both are re‑trained periodically with fresh verified data.

Scaling and safety
• Use the challenge containers and VMEC++ to ensure reproducibility; exploit hot‑restart to ramp throughput. Keep a trust‑region to avoid surrogate overreach and a sanity checker (no self‑intersection, mode bounds). ￼

⸻

4. Physics/Math verification for the key design choices
   1. PCFM feasibility: If constraints g*i(x)=0 are smooth and we project the learned velocity v*\theta(x,t) onto the tangent space T*x\mathcal{M}=\{v:\nabla g(x)^\top v=0\}, then along the flow x’(t)=\Pi*{T*x\mathcal{M}}v*\theta, we have \frac{d}{dt}g(x(t))=0. Hence feasibility is invariant (up to numerical error). This is exactly what PCFM formalizes. ￼
   2. Why minimizing e*{L\nabla B} helps coil simplicity: e*{L\nabla B} is a normalized inverse gradient scale (roughly e\sim a/L*{\nabla B}), and in QP/QI machines L*{\nabla B}\sim R*0/N*{\mathrm{fp}}. For fixed A=R*0/a and N*{\mathrm{fp}}, the global scaling is fixed; local improvements come from reducing high‑m variations that produce sharp \nabla B modulation. Therefore penalties on high‑m energy and phase‑coherent shaping lower e\_{L\nabla B} without violating QI. ￼
   3. Mirror ratio vs QI residual: QI surfaces (near omnigeneity) reduce bounce‑point variation of B along field lines; thus mirror ratio \Delta tends to drop as QI residual decreases. That makes P2’s constraint set physically coherent (feasible region non‑empty and “smooth”), which benefits both surrogates and BO/PCFM flows. ￼
   4. P1 active‑set reasoning: With objective f(x)=\max\_{\phi} \mathrm{elong}(\phi;x), any small change \delta x only affects f through the argmax set of cross‑sections. Penalizing high‑m modes shrinks curvature spikes that create localized maxima, so the subgradient at the active \phi is reduced; KKT multipliers for (A,\tilde\iota,\bar\delta) then settle to a feasible optimum with lower sup‑elongation.
   5. Multi‑objective exploration: For smooth functionals like W*{\mathrm{MHD}} and \langle \chi*{\nabla r}\rangle, q‑EHVI or Tchebycheff scalarization with epsilon‑constraints has known convergence under mild regularity plus uncertainty‑aware exploration; the physics here satisfies those smoothness assumptions (verified by the dataset’s continuous labels). ￼

⸻

5. Concrete “How‑to” checklist
   1. Reproduce baselines from constellaration notebooks, verify local VMEC++ runs match leaderboard evaluations (identical container). Turn on hot‑restart where possible. ￼
   2. Train v0 surrogates (multi‑head MLP or small Transformer on Fourier features + physics features). Add quantile heads; calibrate with conformal intervals. Split by field period and symmetry if needed. ￼
   3. P1: Constrained BO → PCFM manifold sampling → short CMA‑ES polish. Penalize high‑m energy.
   4. P2: PCFM sampler with hard QIres and \Delta caps → PBFM density shaping toward low e\_{L\nabla B} → short CMA‑ES polish.
   5. P3: Two‑stage Pareto (q‑EHVI + feasibility classifiers) with diversity‑aware clustering before high‑fidelity VMEC++.
   6. ASE agents:
      • Retriever/Planner (MLE‑STAR): fetch ideas for model components and acquisition designs; run targeted ablations on one component per loop. ￼
      • Engineer: execute, auto‑debug, adjust VMEC++ settings if runs fail (ASl‑ARCH style). ￼
      • Analyst: update surrogates; write short factual memos into cognition DB. ￼
      • Ensembler: combine proposal streams with an LLM‑planned schedule (weights by recent success + novelty). ￼
   7. Safety/validity gates: geometry sanity checks, constraint screening via calibrated classifiers; always re‑score top candidates at max fidelity before submitting.

⸻

6. What to expect and why we should beat the leaderboard
   • The database gives you an unusually strong prior: high‑quality labeled equilibria in the same parameterization the leaderboard uses. That lets surrogates be predictive enough for PCFM/PBFM to propose feasible‑by‑construction designs—a major time saver. ￼
   • Agentic refinement targets the parts of the pipeline that actually matter (e.g., constraint projection accuracy, acquisition tradeoffs), as shown to be effective in MLE‑STAR. The closed‑loop research pattern in ASI‑ARCH argues for linear gains with compute when the system is designed to learn from its own history; we’ll scale width (diverse generators) and depth (more VFEC evaluations) accordingly.

⸻

Notes on the uploaded files
• I directly used the ConStellaration dataset paper (objectives/constraints), the equilibria database paper (parameterization & correlations), the Landreman–Charkiw lectures (physics heuristics), and the two agent papers (MLE‑STAR, ASI‑ARCH).
• Two markdown uploads (2412.01786v2.md, 2507.12681v1.md) weren’t retrievable via my document search index, so I didn’t quote them. If you want me to incorporate them, I can parse those explicitly on your next message.

⸻

References & sources
• Challenge announcement and repos: Hugging Face blog; Proxima press note; Proxima constellaration and VMEC++ GitHub repos. ￼
• Dataset & physics definitions: ConStellaration dataset paper and equilibria database paper; Landreman–Charkiw stellarator optimization lectures.
• Agent systems: MLE‑STAR (Search + Targeted Refinement); ASI‑ARCH (closed‑loop science at scale).
• Generative samplers for constraints: PBFM (FM meets PDEs) and PCFM (constraint‑manifold FM).

⸻

Final word

The combination of (i) constraint‑preserving generative sampling (PCFM/PBFM), (ii) physics‑informed surrogates trained on the official database, (iii) local polishers (CMA‑ES/BO) with high‑m sparsity priors, and (iv) an agentic research loop (MLE‑STAR + ASI‑ARCH) gives us the right ingredients to systematically climb the leaderboards in P1–P3. The math backs the moves (constraint invariance; e\_{L\nabla B} scaling; QI/mirror coupling), and the engineering choices are aligned with the official toolchain. This is the shortest path to strong results now, and a scalable path to self‑improving discovery thereafter.

Note on LLM agents (scope)
- The current codebase does not integrate LLMs. The “agent” is a deterministic pipeline (propose → evaluate → select) with ablations, novelty, MF gating, constraint hooks, and baseline optimizers. LLM‑driven planner/orchestrator roles remain a research direction and should interface by emitting ablation specs consumable by `constelx ablate run`, with caching and reproducibility.

For a detailed LLM model portfolio and role mapping, see `docs/AGENTS_LLM.md`. This strategy keeps LLMs optional; default CLI paths are deterministic and fully reproducible.

Below is a practical, production‑oriented plan to (i) pick LLMs for your agents (with current, verifiable sources), (ii) give your GPT‑5 coding agent a concrete implementation blueprint end‑to‑end, (iii) map that blueprint to your Apple M3 Max vs. cloud, and (iv) turn the whole thing into a self‑improving system that reliably climbs—and stays at the top of—the ConStellaration leaderboard.

I weave in the physics that matters (VMEC/DESC equilibria, Boozer-space metrics, QS/QH/QI proxies, ALM and constraints) and draw on the two agent papers you uploaded—MLE‑STAR and ASI‑ARCH—translating their ideas into a stellarator‑design context with explicit checks and math. Where I cite those files, I use the inline markers you asked me to use. For challenge specifics and forward solvers I also cite the official Proxima/Hugging Face and VMEC++ sources.

⸻

1. What LLMs should we use for the agents? (SOTA options, by role)

Below I group models by role in the system. In practice, you’ll mix a frontier “planner/orchestrator” with a cost‑effective “coder,” plus small open‑weights for on‑device tasks and ablations. I list top picks first in each category.

A. Planner / Scientist / Orchestrator (deep reasoning, long-horizon planning)
• GPT‑5 (OpenAI) – frontier generalist with improved long‑thinking and strong coding; ideal as the global planner and final reviewer. Official launch pages emphasize deep reasoning and “thinks longer when needed.” ￼
• OpenAI o3 (and o3‑mini) – dedicated reasoning series; o3 sets SOTA on hard benchmarks; o3‑mini gives great cost/latency trade‑offs for inner loops. ￼
• Gemini 2.5 Pro (Google DeepMind) – “thinking built‑in,” strong on coding/agents, huge context; excellent as a co‑planner and cross‑checker, especially when you want multi‑modal scrutiny (plots, Boozer maps). ￼ ￼ ￼
• Claude Sonnet 4 / Opus 4 (Anthropic) – very strong reasoning + deliberate “thinking time” control; Anthropic positions Opus 4 as a top coding model, Sonnet 4 for general use. Good as a second opinion. ￼
• xAI Grok‑3 (“Think” / “Big Brain”) – a capable reasoning family; useful as a diversity check and additional ensemble vote. ￼

B. Coding Agent (robust code synthesis, debugging, tool use)
• GPT‑5 – OpenAI explicitly calls out stronger end‑to‑end code and debugging; use as primary coding agent when budget allows. ￼
• Claude Opus 4 / Sonnet 4 – Anthropic markets Opus as “world’s best coding model”; Sonnet 4 is a cost‑effective alternative. Great at long refactors and error‑driven debugging loops. ￼
• Gemini 2.5 Pro – also “best for coding and complex tasks” per DeepMind’s model page; especially handy for JavaScript/visualization and multi‑modal tooling. ￼
• DeepSeek‑V3 (open MoE) – strong open‑weights option for heavy offline code search/generation; practical for local or dedicated GPU instances. ￼
• Qwen2.5‑Coder (open) – efficient coder models (1.5B–32B) that fine‑tune well for domain‑specific scaffolds (e.g., DESC/VMEC wrappers). ￼
• StarCoder2 (open) – reliable OSS coding backbone with mature tooling. ￼

C. Small open‑weights (local ablations, privacy‑sensitive steps on your M3)
• Phi‑4 / Phi‑4‑reasoning (Microsoft, open‑weights) – 14B “small language model” family with surprisingly strong reasoning; perfect for on‑device linting, doc synthesis, or light planning. ￼ ￼
• Llama 3.2 / Code Llama (Meta, open) – dependable OSS for utility prompts, smaller agents, and offline eval harnesses. ￼

Why a portfolio? Agent systems benefit from model diversity. Use GPT‑5 as the primary planner & coder, back it with Gemini 2.5 Pro and Claude Sonnet/Opus as reviewers/ensemblers, and run Phi‑4 / StarCoder2 locally for fast inner loops.

⸻

2. Implementation plan for your GPT‑5 coding agent (end‑to‑end)

Below is a drop‑in blueprint your GPT‑5 coding agent can follow. It is modular, testable, and maps cleanly onto the ConStellaration challenge tasks and metrics. I explicitly integrate ideas from MLE‑STAR (search + targeted refinement + ensemble) and ASI‑ARCH (researcher/engineer/analyst loop, novelty checks, fitness) and the physics/optimization methods from the challenge docs.

0. Refs the agent should internalize (for prompts & guardrails)
   • Challenge spec + baselines (tasks, metrics, run‑time): three problems (Geometric; “QI simple‑to‑build”; Multi‑objective QI), baseline Augmented Lagrangian + NGOpt, compute notes (e.g., typical VMEC++ and DESC walltimes), and data prior (PCA + classifier/GMM prior). ￼
   • Stellarator optimization fundamentals (boundary → interior uniqueness; QA/QH/QI; Boozer‑space QS surrogate; turbulence proxies; 2‑stage optimization: plasma boundary then coils). ￼
   • Coil optimization via augmented Lagrangian (clear handling of squared‑flux error + geometric constraints with ALM). ￼
   • MLE‑STAR methodology (search to build initial solution; ablation‑guided targeted code‑block refinement; ensemble strategy; data‑leakage and data‑usage checkers). ￼
   • ASI‑ARCH methodology (researcher/engineer/analyst roles; composite fitness = quantitative scores + LLM‑judge; novelty and O(n²) sanity checks; exploration→verification). ￼

1. Repo layout

constellaration/
env/ # conda/mamba envs (macOS, linux CUDA)
src/
physics/
equilibria.py # VMEC++ & DESC wrappers; resolution ladders
booz.py # Boozer transform + QS/QI residuals
metrics.py # A, delta, kappa, iota/Nfp, elongation, f_QS, e_L∇B, proxies
objectives/
problem1_geom.py
problem2_qi_simple.py
problem3_qi_mo.py # multi-objective w/ HV or Chebyshev scalarization
penalties.py # ALM penalties, filters, scaling
optimize/
alm_ngopt.py # baseline reproduction (fair starting point)
grad_trust.py # gradient/adjoint if DESC; trust-region mfidelity
seeds_prior.py # PCA+RF+GMM prior (plus flows) from dataset (baseline idea)
ensemble.py # candidate pooling and model ensembling (MLE‑STAR style)
agents/
planner.py # GPT‑5 planner (high-level problem plan)
coder.py # GPT‑5 coder (implements/refactors components)
ablator.py # extracts pipeline blocks, runs ablations (MLE‑STAR)
judge.py # LLM‑judge (novelty, plausibility; ASI‑ARCH)
memory.py # results DB + embedding retriever (ASI‑ARCH “cognition”)
infra/
runspecs.yaml # problem configs, grids, budgets
slurm_k8s.py # cloud orchestrator
tests/
notebooks/
README.md

2. Forward solvers (resolution ladder and fallbacks)
   • DESC (fast fixed‑boundary equilibria, end‑to‑end autodiff) for inner loops, QS/QI exploration, and gradient checks. DESC supports analytic derivatives and AD, useful for trust‑region steps. ￼
   • VMEC++ (Proxima’s from‑scratch C++/Python re‑implementation) for verification and free‑boundary runs when needed. Proxima documents cross‑platform support and targets Mac and Linux; use it for final scoring fidelity or when the task requires VMEC↔Boozer metrics. ￼ ￼

Why this ladder? Multi‑fidelity improves sample‑efficiency: quick DESC inner steps (seconds–minutes) → VMEC++ confirmation (minutes–~hour depending on grid). The challenge docs explicitly note typical run times for DESC vs. VMEC++—use those to cap inner‑loop budgets. ￼

3. Metrics & constraints (one place, one truth)

Implement pure functions that accept an equilibrium and return:
• Geometry: aspect ratio A, average triangularity \bar\delta, max elongation \max\kappa (or an elongation proxy), curvature limits, coil–surface distances if coil stage is used. (Problem‑1 objective/constraints.) ￼
• Transform: \iota/N*{\rm fp} constraints. ￼
• Quasi‑symmetry/isodynamic proxies: Boozer‑space residual f*{\rm QS} for QA/QH and QI residuals; use standard Boozer angles B(s,\theta-N\phi) forms as surrogate objectives/constraints. ￼
• “Simple‑to‑build” proxy: e*{L\nabla B} term and bounds on aspect/triangularity/curvature as in the challenge’s “QI simple‑to‑build” problem. (These favor coil simplicity by controlling field‑strength variation and geometry.) ￼
• Stability/turbulence proxies: include magnetic well/Mercier and ballooning proxies where available, plus flux‑surface compression in bad curvature and \chi*{\nabla r} turbulence proxies for the multi‑objective case (the challenge spec lists these).

Physics check: The boundary uniquely determines the interior vacuum field via Laplace with Neumann \mathbf{B}\cdot\mathbf{n}=0 (so emphasis on boundary parameterization is justified). QA/QH/QI patterns in B(\theta,\phi) are the right surrogates for confining trapped particles (drift averages to zero along symmetry contours). ￼

4. Parameterizations
   • Boundary Fourier in cylindrical coordinates R*{mn}, Z*{mn} with field‑period symmetry; include integer choices (NFP, helicity) in a small discrete set tried across multi‑starts. Watch for self‑intersection filters. ￼
   • Axis + near‑axis (advanced): allow an option for near‑axis expansions to seed shapes with good QS/QI behavior, then inflate to a boundary surface.

5. Optimization cores

5a. Baseline reproduction (required for fairness):
Augmented Lagrangian with NGOpt (the challenge baseline) for all three problems. This gives you a credible “day‑1” entry and a yardstick for subsequent gains. ￼

5b. Upgrades beyond baseline:
• (i) ALM + Trust‑Region (DESC‑grad): When DESC gradients are available, run a trust‑region SQP‑like inner loop on a coarse grid (mfidelity), with penalty updates \mu_k \uparrow only when infeasible. Switch to VMEC++ to verify improvements at higher resolution before accepting the step (monotone acceptance rule). This blends the coil‑ALM practice with differentiability. ￼
• (ii) Multi‑start seeds from data priors: Fit a prior over feasible designs using the challenge dataset (the baseline describes PCA → RandomForest feasibility → GMM for generative seeding). Extend it with a small normalizing flow. Use the prior for initialization only, not as a hard constraint. ￼
• (iii) Targeted refinement (MLE‑STAR): After each improvement, run ablation to identify which code block (paramization, solver tolerances, Boozer transform options, constraint scalings, mfidelity grid, penalty schedule) most affects the score; then iterate on just that block for K proposals before moving on. Keep a summary log of tried blocks to avoid cycling. ￼
• (iv) Exploration→verification (ASI‑ARCH): Run small‑budget “exploration” campaigns (DESC, coarse modes) spawning many candidates; promote only those that survive novelty/sanity checks and beat a moving baseline into a verification stage (VMEC++, tighter tolerances). ￼
• (v) Ensemble of solutions (MLE‑STAR): Rather than “best‑of‑N,” form an ensemble over distinct optima: e.g., choose among slightly different NFP/helicity or different penalty trajectories, and combine via committee selection (pick the best per‑constraint feasible candidate) or averaged submissions where allowed. ￼

5c. Discrete choices:
Try a grid over NFP (e.g., 3–5), helical linkages, and Boozer N values consistent with the class (QA/QH/QI). Integer choices are usually hand‑optimized; we make them explicit in multi‑start sweeps. ￼

6. Agents and their prompts (how the LLMs plug in)
   • Planner (GPT‑5): Drafts a run plan per problem (param ranges, budgets, mfidelity schedule, seeds to draw, constraint scaling) and updates it after each batch based on score deltas and constraint slacks. ￼
   • Coder (GPT‑5): Implements/refactors a single block (e.g., penalty schedule, Boozer routine options, DESC resolution scaling); integrates error logs; writes unit tests; enforces a standard “contract” for metrics. (MLE‑STAR shows code‑block‑level iteration beats global rewrites.) ￼
   • Ablator (GPT‑5 or Claude): Auto‑builds ablation scripts; executes; summarizes the most sensitive block and proposes 3–5 targeted plans. ￼
   • Judge (LLM‑as‑Judge per ASI‑ARCH): Scores qualitative aspects: novelty (embedding and text check vs archive), plausibility (physics heuristics), and code sanity (e.g., rejects O(n²)–like pathological transforms, missing masks). Its score is auxiliary in a composite fitness used only to prioritize which candidates to verify—the official metric always rules. ￼
   • Memory (ASI‑ARCH “cognition base”): Store distilled “insights” from every run (what setting improved which metric) to bias future proposals; couple to quick retrieval so the planner conditions on recently effective tactics. ￼

7. Safeguards and correctness checks
   • Leakage/usage checkers (from MLE‑STAR) adapted to physics: forbid looking at test targets; ensure all provided data (e.g., required files, geometry) is actually used; in our case: ensure Boozer metrics are computed from the current equilibrium, not cached across shapes. ￼
   • Feasibility filters: reject self‑intersecting boundaries; enforce bounds on A,\bar\delta, curvature; fast checks before expensive equilibria.
   • Sanity checks (ASI‑ARCH): novelty check vs. top‑K archive; O(complexity) checks in code; early‑terminate bad trainings/equilibria (divergence, impossible \iota). ￼

8. Multi‑objective handling (Problem 3)
   • Use \epsilon-constraint or Chebyshev scalarization with rotating weights; log the hypervolume indicator vs. the provided reference point. Keep 4–6 diverse Pareto candidates and let the ensemble pick the one that dominates under the official validator.

9. Coil stage (optional but valuable for “simple‑to‑build”)
   • If coils are part of the scoring or tie‑breaks: run current‑potential or filament coil optimization with Augmented Lagrangian constraints on squared‑flux error, coil curvature, coil–coil and coil–plasma distances—exactly the formulation the coil ALM paper advocates. This locks buildability into the geometry pipeline. ￼

10. Observability & reproducibility
    • Log every run (inputs, grids, seeds, metrics, constraints, walltime) to a lightweight DB + MLFlow/W&B. Keep deterministic seeds for promoted candidates; record solver tolerances.

⸻

3. Compute plan: your Apple M3 Max vs. cloud

Your M3 Max (36 GB) is excellent for development, DESC inner loops, Boozer transforms, and small optimization sweeps:
• Local (macOS, Apple silicon):
• Use conda/mamba env with CPython wheels for DESC and VMEC++ (Proxima says VMEC++ wheels are published and tested on Mac). ￼
• Target DESC at coarse/medium grids; VMEC++ for few verification runs.
• Run Phi‑4 / StarCoder2 locally for agent utility tasks. ￼ ￼

Cloud for breadth & speed (recommended for leaderboard pushes):
• CPU fleets (many VMEC++/DESC in parallel). The ConStellaration docs indicate a typical VMEC++ run can be ~O(1 hr) on ~32 vCPU/32 GB and DESC ~minutes on 32 vCPU/128 GB; scale these horizontally for exploration. Use preemptibles/spot to cut cost and a queue that is checkpoint‑aware. ￼
• GPU nodes (if you adopt learned surrogates/flows or want open‑weights LLMs like DeepSeek‑V3 locally). ￼

⸻

4. How we climb—and stay—at the top of the leaderboard

A. Immediate wins (1–2 weeks) 1. Reproduce the ALM+NGOpt baseline precisely for all three problems (sanity harness). ￼ 2. Add targeted‑refinement (MLE‑STAR): ablate and iterate one block at a time—penalty schedule, mfidelity ladder, Boozer transform options, NFP grid—logging deltas. This reliably yields early improvements. ￼ 3. Turn on exploration→verification (ASI‑ARCH): run many DESC‑coarse inner loops; promote only novel, plausible, and quantitatively superior candidates to VMEC++ verification. ￼ 4. Seed smarter with the data prior (PCA+RF+GMM) from the challenge paper, plus a small normalizing flow; start multi‑starts near inferred feasible basins. ￼

B. Medium‑term gains (2–6 weeks) 5. Trust‑region gradient steps with DESC autodiff; accept steps only when VMEC++ confirms at finer grids (monotone acceptance). This curbs optimizer “false positives.” ￼ 6. Integer design sweeps (NFP, helicity) with small budgets and strict novelty: keep the best per‑class and ensemble. ￼ 7. Coil‑aware penalties in the “simple‑to‑build” QI problem (bound e\_{L\nabla B}, curvature, separations) and—if allowed—coil ALM to demonstrate manufacturability metrics. 8. Ensemble the best diverse solutions (different penalty paths, NFPs) rather than picking a single winner; this raised medal rates in MLE‑STAR. ￼

C. Long‑term, self‑improving production system 9. Institutionalize the ASI‑ARCH loop:
• Researcher (planner) proposes architecture/param changes informed by the memory DB,
• Engineer (coder) implements, runs, and debugs,
• Analyst mines deltas (including parent/sibling comparisons) and writes structured “insights” back to memory. Use a composite fitness = normalized quantitative score improvements + LLM‑judge novelty/plausibility—only to prioritize verification, never to override the official metric. ￼ 10. Knowledge base (“cognition”) of human + AI insights (papers, slide excerpts): store triggers (e.g., “low \iota with tight aspect ratio”) → suggested tactics (e.g., boost certain Z\_{mn} modes). Retrieval feeds the planner. ￼

⸻

Why this should work (physics & optimization checks)
• Boundary → interior uniqueness: For low‑β vacuum fields \nabla \cdot \mathbf{B}=0, \mathbf{J}=\nabla \times \mathbf{B}=0 \Rightarrow \mathbf{B}=\nabla\Phi with Neumann \mathbf{n}\cdot\nabla\Phi=0 on the boundary; solving Laplace + Neumann gives a unique interior field (up to constants). Optimizing the boundary is, therefore, the right handle for many objectives. ￼
• Confinement surrogates (QS/QH/QI): Making B=B(s,\theta-N\phi) in Boozer coordinates yields a conserved quantity for trapped orbits → average drifts cancel → reduced neoclassical transport. Our QS/QI residuals are precisely the surrogates modern design uses. ￼
• “Simple‑to‑build” proxies: Penalizing e\_{L\nabla B}, bounding aspect/triangularity/curvature, and (optionally) coil ALM constraints tie the physics design to real coil complexity. These terms are exactly what the challenge’s QI simple‑to‑build problem encodes, and coil ALM is the state‑of‑practice for manufacturability constraints.
• Augmented Lagrangian + trust region: ALM handles hard constraints robustly without finicky manual scalarization; combining it with gradient‑based trust‑region steps (DESC) and verification (VMEC++) respects nonconvexity and model error—you accept a step only if the higher‑fidelity solver agrees.
• Targeted refinement & ensembles: As MLE‑STAR showed across ML tasks, block‑level ablations to focus exploration and solution ensembling improved medal rates substantially; the same pattern applies here: treat “paramization,” “penalty schedule,” and “resolution” as blocks to refine deeply before moving on, then keep multiple winners. ￼
• Self‑improvement scaling: ASI‑ARCH’s key result is that autonomous discovery scales with compute by closing the “propose‑implement‑analyze” loop with strict fitness, novelty, and sanity checks. Our adaptation constrains those ideas with physics‑truth (VMEC/DESC metrics) while still letting the agents innovate. ￼

⸻

Appendices (operational details your coding agent can execute)

A) Exact problem hooks (from the ConStellaration paper)
• Problem 1 (Geometric): Minimize max elongation subject to bounds on A, \bar\delta, and \iota/N*{\rm fp}. Implement these in objectives/problem1_geom.py. ￼
• Problem 2 (QI simple‑to‑build): Minimize e*{L\nabla B} with constraints on QI residual + geometric bounds; prefer lower coil complexity surrogates. problem2*qi_simple.py. ￼
• Problem 3 (QI multi‑objective): Minimize (-e*{L\nabla B}, A) with constraints (\iota, QI residual, \Delta, W*{\rm MHD}\ge 0, \chi*{\nabla r} cap). Use \epsilon-constraint or Chebyshev scalarization; track hypervolume. problem3_qi_mo.py. ￼

B) Coil stage (when needed)
• Use current‑potential or filament representation; objective: squared‑flux error on the target boundary; constraints: coil curvature \kappa, coil–coil and coil–plasma clearances; optimized by Augmented Lagrangian. physics/coils.py. ￼

C) Agent prompt skeletons (concise)
• Planner prompt: “Given problem‑X objective+constraints and the current archive Δscores, propose a 6‑hour run plan: (i) seeds from prior, (ii) mfidelity ladder (DESC nθ×nζ), (iii) ALM schedule (μ0, update rule), (iv) NFP/helicity grid, (v) ablation target; produce YAML.”
• Coder prompt: “Implement the plan’s block K in module Y; write unit tests; run quick integration on Seed‑S; if crash, fix using logs.”
• Ablator prompt: “Generate an ablation suite toggling one block at a time (penalty schedule, Boozer options, DESC tolerances, NFP); return ranked Δscore and the next refinement target.”
• Judge prompt (ASI‑ARCH): “Score novelty vs. top‑50 (embedding + rationale). Sanity check code complexity & masks. Output {novelty∈[0,1], sanity∈{ok,fail}, rationale}.”

⸻

Where to submit & track scores

Use the official HF Space for submissions/leaderboard and keep local JSON manifests of each entry. ￼

⸻

Model access links you’ll actually use (planner/coder)
• GPT‑5 (planner/coder): official announcements & developer roll‑out. ￼
• OpenAI o3 (backup planner) & o3‑mini (cheap inner loops): ￼
• Gemini 2.5 Pro (co‑planner/coder): ￼ ￼
• Claude Sonnet 4 / Opus 4 (co‑planner/coder): ￼
• DeepSeek‑V3 (open coder): ￼
• Qwen2.5‑Coder (open coder): ￼
• StarCoder2 (open coder): ￼
• VMEC++ (solver): repo + Proxima’s announcement. ￼ ￼
• ConStellaration challenge: press/blog + HF space. ￼ ￼

⸻

Final notes on self‑learning, production grade operation
• Hard separation between exploration and verification. Use DESC for fast exploration; only verified (VMEC++) designs get archived or submitted. This prevents model‑error drift.
• Composite fitness only triages candidates. The leaderboard metric always decides; LLM‑judge influences priority, never acceptance—mirrors ASI‑ARCH’s “fitness + LLM judge” idea but grounded in physics. ￼
• Strict KKT/constraint logs. Every accepted design must come with ALM residuals and constraint slacks; violations are rejected even if the objective improves.
• Diversity maintenance. Keep a candidate pool of top‑K across different NFP/helicity “families” (phylogenetic idea) so we don’t collapse onto a brittle local optimum. ￼
• Ensembling of runs. Submit multiple high‑quality, diverse designs and let the official evaluator pick the best per‑task—this consistently improved win rates in MLE‑STAR. ￼

⸻

(If helpful) A 7‑day execution calendar
• Day 1–2: Stand up envs; reproduce ALM+NGOpt baseline; validate metrics & constraints on small shapes. ￼
• Day 3–4: Wire MLE‑STAR ablation/refinement; add exploration→verification loop; begin smart seeding from dataset prior.
• Day 5: Turn on gradient trust‑region inner steps in DESC + VMEC++ confirmation; sweep small NFP grids. ￼
• Day 6: Ensemble top diverse designs; submit; analyze deltas. ￼
• Day 7: Integrate coil ALM (if relevant to your scoring/constraints) and iterate. ￼

⸻

If you want, I can turn the above into a concrete project scaffold (repo + environment files + first round of unit tests and extension hooks) so your GPT‑5 coding agent can start filling in code immediately.

Below is a practical, production‑oriented plan to (i) pick LLMs for your agents (with current, verifiable sources), (ii) give your GPT‑5 coding agent a concrete implementation blueprint end‑to‑end, (iii) map that blueprint to your Apple M3 Max vs. cloud, and (iv) turn the whole thing into a self‑improving system that reliably climbs—and stays at the top of—the ConStellaration leaderboard.

I weave in the physics that matters (VMEC/DESC equilibria, Boozer-space metrics, QS/QH/QI proxies, ALM and constraints) and draw on the two agent papers you uploaded—MLE‑STAR and ASI‑ARCH—translating their ideas into a stellarator‑design context with explicit checks and math. Where I cite those files, I use the inline markers you asked me to use. For challenge specifics and forward solvers I also cite the official Proxima/Hugging Face and VMEC++ sources.

⸻

1. What LLMs should we use for the agents? (SOTA options, by role)

Below I group models by role in the system. In practice, you’ll mix a frontier “planner/orchestrator” with a cost‑effective “coder,” plus small open‑weights for on‑device tasks and ablations. I list top picks first in each category.

A. Planner / Scientist / Orchestrator (deep reasoning, long-horizon planning)
• GPT‑5 (OpenAI) – frontier generalist with improved long‑thinking and strong coding; ideal as the global planner and final reviewer. Official launch pages emphasize deep reasoning and “thinks longer when needed.” ￼
• OpenAI o3 (and o3‑mini) – dedicated reasoning series; o3 sets SOTA on hard benchmarks; o3‑mini gives great cost/latency trade‑offs for inner loops. ￼
• Gemini 2.5 Pro (Google DeepMind) – “thinking built‑in,” strong on coding/agents, huge context; excellent as a co‑planner and cross‑checker, especially when you want multi‑modal scrutiny (plots, Boozer maps). ￼ ￼ ￼
• Claude Sonnet 4 / Opus 4 (Anthropic) – very strong reasoning + deliberate “thinking time” control; Anthropic positions Opus 4 as a top coding model, Sonnet 4 for general use. Good as a second opinion. ￼
• xAI Grok‑3 (“Think” / “Big Brain”) – a capable reasoning family; useful as a diversity check and additional ensemble vote. ￼

B. Coding Agent (robust code synthesis, debugging, tool use)
• GPT‑5 – OpenAI explicitly calls out stronger end‑to‑end code and debugging; use as primary coding agent when budget allows. ￼
• Claude Opus 4 / Sonnet 4 – Anthropic markets Opus as “world’s best coding model”; Sonnet 4 is a cost‑effective alternative. Great at long refactors and error‑driven debugging loops. ￼
• Gemini 2.5 Pro – also “best for coding and complex tasks” per DeepMind’s model page; especially handy for JavaScript/visualization and multi‑modal tooling. ￼
• DeepSeek‑V3 (open MoE) – strong open‑weights option for heavy offline code search/generation; practical for local or dedicated GPU instances. ￼
• Qwen2.5‑Coder (open) – efficient coder models (1.5B–32B) that fine‑tune well for domain‑specific scaffolds (e.g., DESC/VMEC wrappers). ￼
• StarCoder2 (open) – reliable OSS coding backbone with mature tooling. ￼

C. Small open‑weights (local ablations, privacy‑sensitive steps on your M3)
• Phi‑4 / Phi‑4‑reasoning (Microsoft, open‑weights) – 14B “small language model” family with surprisingly strong reasoning; perfect for on‑device linting, doc synthesis, or light planning. ￼ ￼
• Llama 3.2 / Code Llama (Meta, open) – dependable OSS for utility prompts, smaller agents, and offline eval harnesses. ￼

Why a portfolio? Agent systems benefit from model diversity. Use GPT‑5 as the primary planner & coder, back it with Gemini 2.5 Pro and Claude Sonnet/Opus as reviewers/ensemblers, and run Phi‑4 / StarCoder2 locally for fast inner loops.

⸻

2. Implementation plan for your GPT‑5 coding agent (end‑to‑end)

Below is a drop‑in blueprint your GPT‑5 coding agent can follow. It is modular, testable, and maps cleanly onto the ConStellaration challenge tasks and metrics. I explicitly integrate ideas from MLE‑STAR (search + targeted refinement + ensemble) and ASI‑ARCH (researcher/engineer/analyst loop, novelty checks, fitness) and the physics/optimization methods from the challenge docs.

0. Refs the agent should internalize (for prompts & guardrails)
   • Challenge spec + baselines (tasks, metrics, run‑time): three problems (Geometric; “QI simple‑to‑build”; Multi‑objective QI), baseline Augmented Lagrangian + NGOpt, compute notes (e.g., typical VMEC++ and DESC walltimes), and data prior (PCA + classifier/GMM prior). ￼
   • Stellarator optimization fundamentals (boundary → interior uniqueness; QA/QH/QI; Boozer‑space QS surrogate; turbulence proxies; 2‑stage optimization: plasma boundary then coils). ￼
   • Coil optimization via augmented Lagrangian (clear handling of squared‑flux error + geometric constraints with ALM). ￼
   • MLE‑STAR methodology (search to build initial solution; ablation‑guided targeted code‑block refinement; ensemble strategy; data‑leakage and data‑usage checkers). ￼
   • ASI‑ARCH methodology (researcher/engineer/analyst roles; composite fitness = quantitative scores + LLM‑judge; novelty and O(n²) sanity checks; exploration→verification). ￼

1. Repo layout

constellaration/
env/ # conda/mamba envs (macOS, linux CUDA)
src/
physics/
equilibria.py # VMEC++ & DESC wrappers; resolution ladders
booz.py # Boozer transform + QS/QI residuals
metrics.py # A, delta, kappa, iota/Nfp, elongation, f_QS, e_L∇B, proxies
objectives/
problem1_geom.py
problem2_qi_simple.py
problem3_qi_mo.py # multi-objective w/ HV or Chebyshev scalarization
penalties.py # ALM penalties, filters, scaling
optimize/
alm_ngopt.py # baseline reproduction (fair starting point)
grad_trust.py # gradient/adjoint if DESC; trust-region mfidelity
seeds_prior.py # PCA+RF+GMM prior (plus flows) from dataset (baseline idea)
ensemble.py # candidate pooling and model ensembling (MLE‑STAR style)
agents/
planner.py # GPT‑5 planner (high-level problem plan)
coder.py # GPT‑5 coder (implements/refactors components)
ablator.py # extracts pipeline blocks, runs ablations (MLE‑STAR)
judge.py # LLM‑judge (novelty, plausibility; ASI‑ARCH)
memory.py # results DB + embedding retriever (ASI‑ARCH “cognition”)
infra/
runspecs.yaml # problem configs, grids, budgets
slurm_k8s.py # cloud orchestrator
tests/
notebooks/
README.md

2. Forward solvers (resolution ladder and fallbacks)
   • DESC (fast fixed‑boundary equilibria, end‑to‑end autodiff) for inner loops, QS/QI exploration, and gradient checks. DESC supports analytic derivatives and AD, useful for trust‑region steps. ￼
   • VMEC++ (Proxima’s from‑scratch C++/Python re‑implementation) for verification and free‑boundary runs when needed. Proxima documents cross‑platform support and targets Mac and Linux; use it for final scoring fidelity or when the task requires VMEC↔Boozer metrics. ￼ ￼

Why this ladder? Multi‑fidelity improves sample‑efficiency: quick DESC inner steps (seconds–minutes) → VMEC++ confirmation (minutes–~hour depending on grid). The challenge docs explicitly note typical run times for DESC vs. VMEC++—use those to cap inner‑loop budgets. ￼

3. Metrics & constraints (one place, one truth)

Implement pure functions that accept an equilibrium and return:
• Geometry: aspect ratio A, average triangularity \bar\delta, max elongation \max\kappa (or an elongation proxy), curvature limits, coil–surface distances if coil stage is used. (Problem‑1 objective/constraints.) ￼
• Transform: \iota/N*{\rm fp} constraints. ￼
• Quasi‑symmetry/isodynamic proxies: Boozer‑space residual f*{\rm QS} for QA/QH and QI residuals; use standard Boozer angles B(s,\theta-N\phi) forms as surrogate objectives/constraints. ￼
• “Simple‑to‑build” proxy: e*{L\nabla B} term and bounds on aspect/triangularity/curvature as in the challenge’s “QI simple‑to‑build” problem. (These favor coil simplicity by controlling field‑strength variation and geometry.) ￼
• Stability/turbulence proxies: include magnetic well/Mercier and ballooning proxies where available, plus flux‑surface compression in bad curvature and \chi*{\nabla r} turbulence proxies for the multi‑objective case (the challenge spec lists these).

Physics check: The boundary uniquely determines the interior vacuum field via Laplace with Neumann \mathbf{B}\cdot\mathbf{n}=0 (so emphasis on boundary parameterization is justified). QA/QH/QI patterns in B(\theta,\phi) are the right surrogates for confining trapped particles (drift averages to zero along symmetry contours). ￼

4. Parameterizations
   • Boundary Fourier in cylindrical coordinates R*{mn}, Z*{mn} with field‑period symmetry; include integer choices (NFP, helicity) in a small discrete set tried across multi‑starts. Watch for self‑intersection filters. ￼
   • Axis + near‑axis (advanced): allow an option for near‑axis expansions to seed shapes with good QS/QI behavior, then inflate to a boundary surface.

5. Optimization cores

5a. Baseline reproduction (required for fairness):
Augmented Lagrangian with NGOpt (the challenge baseline) for all three problems. This gives you a credible “day‑1” entry and a yardstick for subsequent gains. ￼

5b. Upgrades beyond baseline:
• (i) ALM + Trust‑Region (DESC‑grad): When DESC gradients are available, run a trust‑region SQP‑like inner loop on a coarse grid (mfidelity), with penalty updates \mu_k \uparrow only when infeasible. Switch to VMEC++ to verify improvements at higher resolution before accepting the step (monotone acceptance rule). This blends the coil‑ALM practice with differentiability. ￼
• (ii) Multi‑start seeds from data priors: Fit a prior over feasible designs using the challenge dataset (the baseline describes PCA → RandomForest feasibility → GMM for generative seeding). Extend it with a small normalizing flow. Use the prior for initialization only, not as a hard constraint. ￼
• (iii) Targeted refinement (MLE‑STAR): After each improvement, run ablation to identify which code block (paramization, solver tolerances, Boozer transform options, constraint scalings, mfidelity grid, penalty schedule) most affects the score; then iterate on just that block for K proposals before moving on. Keep a summary log of tried blocks to avoid cycling. ￼
• (iv) Exploration→verification (ASI‑ARCH): Run small‑budget “exploration” campaigns (DESC, coarse modes) spawning many candidates; promote only those that survive novelty/sanity checks and beat a moving baseline into a verification stage (VMEC++, tighter tolerances). ￼
• (v) Ensemble of solutions (MLE‑STAR): Rather than “best‑of‑N,” form an ensemble over distinct optima: e.g., choose among slightly different NFP/helicity or different penalty trajectories, and combine via committee selection (pick the best per‑constraint feasible candidate) or averaged submissions where allowed. ￼

5c. Discrete choices:
Try a grid over NFP (e.g., 3–5), helical linkages, and Boozer N values consistent with the class (QA/QH/QI). Integer choices are usually hand‑optimized; we make them explicit in multi‑start sweeps. ￼

6. Agents and their prompts (how the LLMs plug in)
   • Planner (GPT‑5): Drafts a run plan per problem (param ranges, budgets, mfidelity schedule, seeds to draw, constraint scaling) and updates it after each batch based on score deltas and constraint slacks. ￼
   • Coder (GPT‑5): Implements/refactors a single block (e.g., penalty schedule, Boozer routine options, DESC resolution scaling); integrates error logs; writes unit tests; enforces a standard “contract” for metrics. (MLE‑STAR shows code‑block‑level iteration beats global rewrites.) ￼
   • Ablator (GPT‑5 or Claude): Auto‑builds ablation scripts; executes; summarizes the most sensitive block and proposes 3–5 targeted plans. ￼
   • Judge (LLM‑as‑Judge per ASI‑ARCH): Scores qualitative aspects: novelty (embedding and text check vs archive), plausibility (physics heuristics), and code sanity (e.g., rejects O(n²)–like pathological transforms, missing masks). Its score is auxiliary in a composite fitness used only to prioritize which candidates to verify—the official metric always rules. ￼
   • Memory (ASI‑ARCH “cognition base”): Store distilled “insights” from every run (what setting improved which metric) to bias future proposals; couple to quick retrieval so the planner conditions on recently effective tactics. ￼

7. Safeguards and correctness checks
   • Leakage/usage checkers (from MLE‑STAR) adapted to physics: forbid looking at test targets; ensure all provided data (e.g., required files, geometry) is actually used; in our case: ensure Boozer metrics are computed from the current equilibrium, not cached across shapes. ￼
   • Feasibility filters: reject self‑intersecting boundaries; enforce bounds on A,\bar\delta, curvature; fast checks before expensive equilibria.
   • Sanity checks (ASI‑ARCH): novelty check vs. top‑K archive; O(complexity) checks in code; early‑terminate bad trainings/equilibria (divergence, impossible \iota). ￼

8. Multi‑objective handling (Problem 3)
   • Use \epsilon-constraint or Chebyshev scalarization with rotating weights; log the hypervolume indicator vs. the provided reference point. Keep 4–6 diverse Pareto candidates and let the ensemble pick the one that dominates under the official validator.

9. Coil stage (optional but valuable for “simple‑to‑build”)
   • If coils are part of the scoring or tie‑breaks: run current‑potential or filament coil optimization with Augmented Lagrangian constraints on squared‑flux error, coil curvature, coil–coil and coil–plasma distances—exactly the formulation the coil ALM paper advocates. This locks buildability into the geometry pipeline. ￼

10. Observability & reproducibility
    • Log every run (inputs, grids, seeds, metrics, constraints, walltime) to a lightweight DB + MLFlow/W&B. Keep deterministic seeds for promoted candidates; record solver tolerances.

⸻

3. Compute plan: your Apple M3 Max vs. cloud

Your M3 Max (36 GB) is excellent for development, DESC inner loops, Boozer transforms, and small optimization sweeps:
• Local (macOS, Apple silicon):
• Use conda/mamba env with CPython wheels for DESC and VMEC++ (Proxima says VMEC++ wheels are published and tested on Mac). ￼
• Target DESC at coarse/medium grids; VMEC++ for few verification runs.
• Run Phi‑4 / StarCoder2 locally for agent utility tasks. ￼ ￼

Cloud for breadth & speed (recommended for leaderboard pushes):
• CPU fleets (many VMEC++/DESC in parallel). The ConStellaration docs indicate a typical VMEC++ run can be ~O(1 hr) on ~32 vCPU/32 GB and DESC ~minutes on 32 vCPU/128 GB; scale these horizontally for exploration. Use preemptibles/spot to cut cost and a queue that is checkpoint‑aware. ￼
• GPU nodes (if you adopt learned surrogates/flows or want open‑weights LLMs like DeepSeek‑V3 locally). ￼

⸻

4. How we climb—and stay—at the top of the leaderboard

A. Immediate wins (1–2 weeks) 1. Reproduce the ALM+NGOpt baseline precisely for all three problems (sanity harness). ￼ 2. Add targeted‑refinement (MLE‑STAR): ablate and iterate one block at a time—penalty schedule, mfidelity ladder, Boozer transform options, NFP grid—logging deltas. This reliably yields early improvements. ￼ 3. Turn on exploration→verification (ASI‑ARCH): run many DESC‑coarse inner loops; promote only novel, plausible, and quantitatively superior candidates to VMEC++ verification. ￼ 4. Seed smarter with the data prior (PCA+RF+GMM) from the challenge paper, plus a small normalizing flow; start multi‑starts near inferred feasible basins. ￼

B. Medium‑term gains (2–6 weeks) 5. Trust‑region gradient steps with DESC autodiff; accept steps only when VMEC++ confirms at finer grids (monotone acceptance). This curbs optimizer “false positives.” ￼ 6. Integer design sweeps (NFP, helicity) with small budgets and strict novelty: keep the best per‑class and ensemble. ￼ 7. Coil‑aware penalties in the “simple‑to‑build” QI problem (bound e\_{L\nabla B}, curvature, separations) and—if allowed—coil ALM to demonstrate manufacturability metrics. 8. Ensemble the best diverse solutions (different penalty paths, NFPs) rather than picking a single winner; this raised medal rates in MLE‑STAR. ￼

C. Long‑term, self‑improving production system 9. Institutionalize the ASI‑ARCH loop:
• Researcher (planner) proposes architecture/param changes informed by the memory DB,
• Engineer (coder) implements, runs, and debugs,
• Analyst mines deltas (including parent/sibling comparisons) and writes structured “insights” back to memory. Use a composite fitness = normalized quantitative score improvements + LLM‑judge novelty/plausibility—only to prioritize verification, never to override the official metric. ￼ 10. Knowledge base (“cognition”) of human + AI insights (papers, slide excerpts): store triggers (e.g., “low \iota with tight aspect ratio”) → suggested tactics (e.g., boost certain Z\_{mn} modes). Retrieval feeds the planner. ￼

⸻

Why this should work (physics & optimization checks)
• Boundary → interior uniqueness: For low‑β vacuum fields \nabla \cdot \mathbf{B}=0, \mathbf{J}=\nabla \times \mathbf{B}=0 \Rightarrow \mathbf{B}=\nabla\Phi with Neumann \mathbf{n}\cdot\nabla\Phi=0 on the boundary; solving Laplace + Neumann gives a unique interior field (up to constants). Optimizing the boundary is, therefore, the right handle for many objectives. ￼
• Confinement surrogates (QS/QH/QI): Making B=B(s,\theta-N\phi) in Boozer coordinates yields a conserved quantity for trapped orbits → average drifts cancel → reduced neoclassical transport. Our QS/QI residuals are precisely the surrogates modern design uses. ￼
• “Simple‑to‑build” proxies: Penalizing e\_{L\nabla B}, bounding aspect/triangularity/curvature, and (optionally) coil ALM constraints tie the physics design to real coil complexity. These terms are exactly what the challenge’s QI simple‑to‑build problem encodes, and coil ALM is the state‑of‑practice for manufacturability constraints.
• Augmented Lagrangian + trust region: ALM handles hard constraints robustly without finicky manual scalarization; combining it with gradient‑based trust‑region steps (DESC) and verification (VMEC++) respects nonconvexity and model error—you accept a step only if the higher‑fidelity solver agrees.
• Targeted refinement & ensembles: As MLE‑STAR showed across ML tasks, block‑level ablations to focus exploration and solution ensembling improved medal rates substantially; the same pattern applies here: treat “paramization,” “penalty schedule,” and “resolution” as blocks to refine deeply before moving on, then keep multiple winners. ￼
• Self‑improvement scaling: ASI‑ARCH’s key result is that autonomous discovery scales with compute by closing the “propose‑implement‑analyze” loop with strict fitness, novelty, and sanity checks. Our adaptation constrains those ideas with physics‑truth (VMEC/DESC metrics) while still letting the agents innovate. ￼

⸻

Appendices (operational details your coding agent can execute)

A) Exact problem hooks (from the ConStellaration paper)
• Problem 1 (Geometric): Minimize max elongation subject to bounds on A, \bar\delta, and \iota/N*{\rm fp}. Implement these in objectives/problem1_geom.py. ￼
• Problem 2 (QI simple‑to‑build): Minimize e*{L\nabla B} with constraints on QI residual + geometric bounds; prefer lower coil complexity surrogates. problem2*qi_simple.py. ￼
• Problem 3 (QI multi‑objective): Minimize (-e*{L\nabla B}, A) with constraints (\iota, QI residual, \Delta, W*{\rm MHD}\ge 0, \chi*{\nabla r} cap). Use \epsilon-constraint or Chebyshev scalarization; track hypervolume. problem3_qi_mo.py. ￼

B) Coil stage (when needed)
• Use current‑potential or filament representation; objective: squared‑flux error on the target boundary; constraints: coil curvature \kappa, coil–coil and coil–plasma clearances; optimized by Augmented Lagrangian. physics/coils.py. ￼

C) Agent prompt skeletons (concise)
• Planner prompt: “Given problem‑X objective+constraints and the current archive Δscores, propose a 6‑hour run plan: (i) seeds from prior, (ii) mfidelity ladder (DESC nθ×nζ), (iii) ALM schedule (μ0, update rule), (iv) NFP/helicity grid, (v) ablation target; produce YAML.”
• Coder prompt: “Implement the plan’s block K in module Y; write unit tests; run quick integration on Seed‑S; if crash, fix using logs.”
• Ablator prompt: “Generate an ablation suite toggling one block at a time (penalty schedule, Boozer options, DESC tolerances, NFP); return ranked Δscore and the next refinement target.”
• Judge prompt (ASI‑ARCH): “Score novelty vs. top‑50 (embedding + rationale). Sanity check code complexity & masks. Output {novelty∈[0,1], sanity∈{ok,fail}, rationale}.”

⸻

Where to submit & track scores

Use the official HF Space for submissions/leaderboard and keep local JSON manifests of each entry. ￼

⸻

Model access links you’ll actually use (planner/coder)
• GPT‑5 (planner/coder): official announcements & developer roll‑out. ￼
• OpenAI o3 (backup planner) & o3‑mini (cheap inner loops): ￼
• Gemini 2.5 Pro (co‑planner/coder): ￼ ￼
• Claude Sonnet 4 / Opus 4 (co‑planner/coder): ￼
• DeepSeek‑V3 (open coder): ￼
• Qwen2.5‑Coder (open coder): ￼
• StarCoder2 (open coder): ￼
• VMEC++ (solver): repo + Proxima’s announcement. ￼ ￼
• ConStellaration challenge: press/blog + HF space. ￼ ￼

⸻

Final notes on self‑learning, production grade operation
• Hard separation between exploration and verification. Use DESC for fast exploration; only verified (VMEC++) designs get archived or submitted. This prevents model‑error drift.
• Composite fitness only triages candidates. The leaderboard metric always decides; LLM‑judge influences priority, never acceptance—mirrors ASI‑ARCH’s “fitness + LLM judge” idea but grounded in physics. ￼
• Strict KKT/constraint logs. Every accepted design must come with ALM residuals and constraint slacks; violations are rejected even if the objective improves.
• Diversity maintenance. Keep a candidate pool of top‑K across different NFP/helicity “families” (phylogenetic idea) so we don’t collapse onto a brittle local optimum. ￼
• Ensembling of runs. Submit multiple high‑quality, diverse designs and let the official evaluator pick the best per‑task—this consistently improved win rates in MLE‑STAR. ￼

⸻

(If helpful) A 7‑day execution calendar
• Day 1–2: Stand up envs; reproduce ALM+NGOpt baseline; validate metrics & constraints on small shapes. ￼
• Day 3–4: Wire MLE‑STAR ablation/refinement; add exploration→verification loop; begin smart seeding from dataset prior.
• Day 5: Turn on gradient trust‑region inner steps in DESC + VMEC++ confirmation; sweep small NFP grids. ￼
• Day 6: Ensemble top diverse designs; submit; analyze deltas. ￼
• Day 7: Integrate coil ALM (if relevant to your scoring/constraints) and iterate. ￼

⸻

If you want, I can turn the above into a concrete project scaffold (repo + environment files + first round of unit tests and extension hooks) so your GPT‑5 coding agent can start filling in code immediately.

Yes—this paper is packed with techniques we can use right away. Here’s a practitioner‑oriented extraction of the most useful ideas and how to apply them to ConStellaration (and beyond).

⸻

What the paper actually does (and why it matters)

1. Replace hand‑tuned weights with an Augmented Lagrangian (AL) scheme.
   Instead of the usual weighted sum of many coil engineering “objectives,” the authors formulate all physics/engineering requirements as constraints and solve a sequence of unconstrained problems with an augmented Lagrangian:
   \min_x\; L_A(x,\lambda,\mu)=f(x)\;-\;\lambda^\top c(x)\;+\;\tfrac12\lVert \sqrt{\mu}\circ c(x)\rVert_2^2,
   updating multipliers \lambda and penalties \mu adaptively. Crucially, they set the objective \(f(x)\equiv 0 and push even the “squared flux” term (normally the main objective) into the constraints with a tolerance\*\*, so the optimizer stops over‑fitting field error once it’s “good enough” and spends effort improving buildability and forces. This eliminates manual weight sweeps and usually finds better trade‑offs. ￼

2. Inequality constraints via “positive‑part” transformation.
   Every inequality \(g(x)\le 0\) is rewritten as c(x)=\max(g(x)-g\_{\text{target}},0), letting AL penalize only violations; satisfied constraints go silent and free up optimization effort. ￼

3. Filamentary coil parameterization that’s AD‑friendly.
   Each coil curve is a truncated Fourier series in arc length; currents are included as decision variables. Gradients of all constraint terms are computed with a mix of analytic derivatives + automatic differentiation and optimized with L‑BFGS‑B inside each AL inner step—giving fast, stable convergence on hundreds of DoFs. ￼

4. Treat field accuracy as an inequality with a realistic tolerance.
   They use the normalized squared flux f*{SF} as a constraint not a “chase‑to‑zero” objective—e.g., enforce f*{SF}\lesssim 10^{-6}. This reflects reality (you’ll never achieve exactly zero error because of manufacturing and integration error fields) and prevents wasting iterations “over‑polishing” field fit at the expense of engineering metrics. ￼

5. Buildability & safety encoded as physics‑meaningful constraints with natural bounds.
   They define a compact set of coil complexity metrics—each with clear physical meaning and reactor‑scale target bounds (Table I of the paper):
   • Min coil‑surface clearance d*{cs}\gtrsim 1.3\,\text{m} (blanket/divertor space).
   • Min coil‑coil clearance d*{cc}\gtrsim 0.7\,\text{m}.
   • Total coil length L \lesssim 150\!-\!200\,\text{m}.
   • Max curvature \kappa\_{\max}\lesssim 1\,\text{m}^{-1} and mean‑squared curvature (MSC) \lesssim 0.5\,\text{m}^{-1}.
   • Gauss linking number = 0 (avoid interlinked coils).
   • Max Lorentz force per turn \lesssim 0.5\,\text{MN/m} (HTS tolerance proxy).
   All are written as AL constraints (with the positive‑part form), so once they’re satisfied they stop dominating the search. ￼

6. Pareto fronts “for free.”
   By sweeping just a few physical thresholds (e.g., allowable force levels) while AL handles the rest, they trace out or improve existing Pareto fronts (field error vs. forces) with far fewer runs than weight‑scanning approaches; they report ~40 minutes per run on a single core for their QA case studies. ￼

7. Practical insights you can use as design priors.
   • Treating f\_{SF} as a constraint lets the optimizer trade “surplus” field fidelity for simpler coils.
   • Fewer coils per half‑field period can be feasible at fixed total length, improving port/access but potentially requiring more HTS turns—use AL to balance these.
   • Topological guards (linking number) matter; otherwise local optima can “cheat” by interlinking coils. ￼

⸻

How to port the techniques into the ConStellaration challenge

ConStellaration is a stage‑1 (plasma‑boundary) optimization benchmark with tasks like geometric minimization, “simple‑to‑build QI,” and MHD‑stable QI; they already show ALM‑Nevergrad baselines that work when first‑order methods struggle. The coil‑paper’s AL recipe suggests several upgrades and cross‑overs. ￼

A) Use “constraints‑as‑first‑class citizens” with realistic tolerances.
For each benchmark, push everything except the single score‑defining target into inequality constraints with explicit tolerances, e.g.:
• Geometric task:
• Objective: minimize max elongation \epsilon*{\max}.
• Constraints: aspect ratio A\le A^, triangularity \bar\delta\le \bar\delta^, edge transform per NFP \tilde\iota\ge \tilde\iota^\*.
Implement AL exactly as in the coil paper, so that once the constraints are within tolerance they stop “pulling.” ￼
• Simple‑to‑build QI:
• Objective: minimize normalized gradient‑scale length \tilde L*{\nabla B} (a coil‑simplicity proxy).
• Constraints: QI residual, mirror ratio \Delta, target \tilde\iota, aspect ratio A, max elongation \epsilon*{\max}.
Use inequality transforms c_i=\max(g_i-g_i^\*,0) from the coil paper; adapt their “stop polishing when good enough” idea by moving some physics targets into constraints (e.g., “QI residual \le threshold”) instead of objectives. That should free search effort to reduce \tilde L*{\nabla B}.
• MHD‑stable QI:
• Multi‑objective (compactness vs. coil simplicity). Decompose to single‑objective runs by treating A as a constraint and scan only the bound A^\* (like they scan force thresholds), letting AL shape the rest. This mirrors their “few bound sweeps → Pareto front” trick. ￼

B) Copy their inner–outer algorithmic structure.
• Inner loop: L‑BFGS‑B minimize L*A(\cdot,\lambda,\mu) to a gradient tolerance \omega.
• Outer loop: If constraint max‑violation \|c(x)\|\infty drops below \eta: update multipliers \lambda\leftarrow\lambda-\mu\circ c(x); else inflate penalties \mu\leftarrow \tau\mu (\tau>1).
Exactly the same logic they used for coils transfers to stage‑1 because your boundary parameterization (Fourier R{mn},Z*{mn}) is smooth and VMEC++/DESC provide differentiable metrics.

C) Borrow their “physics‑meaningful bounds not weights” philosophy.
Pick tolerances/targets close to ConStellaration’s evaluation (e.g., QI residual, mirror ratio, \langle\chi\_{\nabla r}\rangle cap for turbulence proxy), not arbitrary weights. This makes solutions robust and directly comparable on the leaderboard’s scoring code. ￼

D) Tie stage‑1 to stage‑2 (coil) feasibility with proxy constraints.
Although ConStellaration is stage‑1, you can add soft constraints that bias the boundary toward coil‑friendly geometry (e.g., penalize high |\nabla B| variations in Boozer angles, or directly minimize \tilde L\_{\nabla B} as they propose). The coil paper’s natural bounds (curvature/clearances/forces) are excellent priors to calibrate your proxies or to set feasibility windows for subsequent coil optimization.

⸻

Concrete “do‑this‑now” checklist 1. Switch your loss to AL form and move “physics targets” into constraints with tolerances (start with the paper’s inequality transform). Use L‑BFGS‑B for the inner solve; carry (\lambda,\mu) across runs of the same target to warm‑start. ￼ 2. Set tolerances realistically (e.g., treat QI residual, mirror, \tilde\iota like the coil paper treats f\_{SF}): once satisfied, stop tightening—let the solver simplify geometry. ￼ 3. Generate Pareto curves by scanning only one bound (e.g., compactness A^\*) and leave the rest to AL. This replicates their efficient Pareto tracing without massive weight scans. 4. Adopt differentiable metrics end‑to‑end (VMEC++ or DESC) so gradients are accurate; the paper’s mixed analytic/AD chain‑rule implementation is a blueprint. ￼ 5. Keep topological safety rails (no interlinked features) via simple scalar constraints, just like their Gauss linking number—prevents “cheating” minima. ￼

⸻

Subtleties & pitfalls (their experience helps)
• Don’t chase absolute zero field error. In practice, once you’re below the tolerance, “slack” is better spent on curvature/forces/clearances; the AL setup enforces this automatically. ￼
• Penalty growth vs. multiplier updates. If violations don’t fall, grow \mu rather than over‑zealously changing \lambda. The paper’s update rules (and convergence criteria using \omega*{\text{tol}},\eta*{\text{tol}}) are good defaults. ￼
• Guard against degenerate solutions. Maintain lower/upper bounds on all geometry metrics even if you’re not “actively” optimizing them—this prevents pathological shapes (they stress this). ￼

⸻

Why this should help on the leaderboard

ConStellaration’s baselines already show that AL beats naïve trust‑region and gradient‑free methods when evaluated fairly (e.g., only ALM‑NGOpt obtained feasible solutions in their single‑objective tests). The coil paper gives you a clean, physics‑motivated AL variant and concrete tolerance‑first design that (i) avoids hyperparameter twiddling, (ii) tends to find better coil‑friendly shapes at the same physics quality, and (iii) can trace compactness–simplicity Pareto fronts efficiently—all items ConStellaration scores directly. ￼

⸻

One page to copy into your codebase (algorithm sketch)

Augmented Lagrangian loop (as in the paper):
• Initialize x*0 (boundary Fourier coefficients), \lambda_0\sim U[0,1], \mu_0>0, tolerances \omega*{\text{tol}},\eta*{\text{tol}}, growth \tau>1.
• Repeat: 1. x*{p+1} \leftarrow \arg\min*x L_A(x,\lambda_p,\mu_p) with L‑BFGS‑B to gradient tol \omega_p. 2. If \|c(x*{p+1})\|\infty<\eta*p: \*\*\lambda{p+1}\leftarrow \lambda_p-\mu_p\circ c(x*{p+1}), keep \mu fixed\*\*; shrink \omega,\eta.
Else: \mu*{p+1}\leftarrow \tau\,\mu_p; keep \lambda fixed; reset/shrink \omega,\eta.
• Stop when \|\nabla L_A(x*{p+1},\lambda*{p+1},\mu*{p+1})\|2\le \omega{\text{tol}} and \|c(x\_{p+1})\|\infty\le \eta{\text{tol}}.
This is verbatim the control logic they used for coils; it translates directly to stage‑1 metrics. ￼

⸻

TL;DR
• Use their AL with constraints‑as‑targets (and a reasonable tolerance on field metrics) to avoid weight tuning and unlock coil/buildability improvements without sacrificing physics.
• Scan one physical bound (e.g., compactness) to generate Pareto fronts quickly.
• Adopt physics‑meaningful bounds as guardrails; keep topology checks; leverage AD + L‑BFGS‑B for speed and stability.

If you want, I can draft a minimal AL optimizer wrapper for VMEC++/DESC that mirrors their update rules and plugs into ConStellaration’s scoring scripts.

Primary source: Augmented Lagrangian methods produce cutting-edge magnetic coils for stellarator fusion reactors (algorithm, metrics, bounds, results). ￼
Challenge reference: ConStellaration dataset & benchmarks (what the leaderboard measures and how AL is used today). ￼
