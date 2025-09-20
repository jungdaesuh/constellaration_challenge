1. What the challenge is (and how you’re scored)

Scope & tooling. The benchmark targets stage‑1 stellarator design: optimize the plasma boundary (Fourier surface in cylindrical R,Z) and score it by running a forward model (VMEC++‑based) that computes ideal‑MHD equilibria and metrics. The project provides a dataset of QI‑like boundaries + equilibria, with evaluation code and a leaderboard. Submissions are JSON boundaries in the repo’s Pydantic schema. ￼ ￼

Three problems. 1. Geometric (single‑objective): minimize max elongation \epsilon*\text{max} at fixed aspect ratio A, edge rotational transform per field period \iota*\text{edge}/N*\text{fp}, and average triangularity \langle\delta\rangle. 2. Simple‑to‑build QI (single‑objective): maximize coil simplicity via normalized magnetic‑field gradient scale length \tilde L*{\nabla B} while enforcing QI residual smallness and bounding mirror ratio \Delta, A, \epsilon*\text{max}, etc. (the metric choice follows Kappel et al. for coil simplicity; QI residual follows Goodman et al.). 3. MHD‑stable QI (multi‑objective): trade off compactness (low A) vs coil simplicity (high \tilde L*{\nabla B}), subject to constraints including vacuum magnetic well W*\text{MHD} and a turbulence proxy based on flux compression in regions of bad curvature \chi*{\nabla r} (evaluated at \rho=0.7). Scoring is hypervolume over feasible points. ￼

Scoring formalism. Single‑objective problems map to a bounded score s(\Theta)\in[0,1] if all normalized constraints \tilde c_i\le \varepsilon, else 0; multi‑objective uses HV over feasible solutions relative to a fixed reference. Baselines (ALM+Nevergrad) are provided and are the only ones that produced feasible solutions in the paper’s study. ￼

What you’re given. The HF dataset contains plasma boundaries (Fourier coefficients), equilibria (“wout”) and metrics; the constellaration Python package (PyPI/GitHub) includes evaluation functions, forward‑model wrappers, and submission helpers. VMEC++ is an open‑source, modernized VMEC with a Python‑first interface. ￼ ￼

Useful physics background: stage-1 optimizes a Fourier boundary in R(\theta,\phi),Z(\theta,\phi); quasi-symmetry / quasi-isodynamic metrics are naturally defined in Boozer coordinates; coil simplicity proxies and MHD/turbulence proxies are standard “stage-1” surrogates. ￼ Implementation tip: `constelx.eval.boozer.compute_boozer_proxies` exposes bounded QS/QI residuals and mirror-ratio proxies so guardrails can run before paying for full VMEC++ solves.

⸻

2. Design implications from the spec (what matters)
   • Representation matters. All scoring assumes the truncated Fourier boundary (stellarator‑symmetric). Proper mode selection, scaling/whitening, and preconditioning of coefficients is crucial for optimizer stability and surrogate learning. ￼
   • Feasibility first. Single‑objective scores collapse to 0 if any constraint is violated; multi‑objective points are excluded if infeasible. This strongly favors constraint‑aware search and trust‑region behavior. ￼
   • Expandable fidelity. VMEC++ can be run at lower resolution during search and at higher fidelity for scoring; exploiting multi‑fidelity is a core efficiency lever. ￼
   • Physics hints. QI often pushes to higher A, large \epsilon\_\text{max}, and larger \Delta; the simple‑to‑build problem pushes back with bounds and a coil simplicity objective—expect tight constraint surfaces. The MHD/turbulence constraints are nontrivial and prone to VMEC convergence failures in certain corners. ￼

⸻

3. End‑to‑end plan to compete

I recommend three coordinated tracks that share one evaluation harness:

Track 0 — Reproduce & harden the baselines (quick win)

Goal: Establish a reliable, parallelized baseline you can trust and iterate on.
• Environment: Build their Docker image or install constellaration + libnetcdf-dev locally; keep a high‑fidelity VMEC++ “scorer” separate from a low‑fidelity “search” model. ￼
• Optimizer: Re‑implement their ALM + Nevergrad (NGOpt) with:
• Coefficient whitening & diagonal preconditioner (exponential decay by mode index), as they did.
• A shrinking trust‑region proximal step in the ALM primal update.
• Multi‑start / multi‑seed runs (e.g., rotating ellipse seeds varied in aspect/phase). ￼
• VMEC settings: Start search with lower N_s, looser force tolerance; re‑score promising candidates at high fidelity. Gate on convergence flags to prevent false positives. ￼
• Deliverables: A “baseline+” that reproduces within paper range (e.g., geometric \sim 0.97 single‑objective score) and submits JSON correctly to the HF Space. ￼

Track 1 — Surrogate‑aided, constraint‑aware optimization (efficiency)

Goal: Cut oracle calls by 5–20× while improving feasibility hit rate.
• Structured surrogate. Train a mode‑aware MLP/Transformer that ingests \{R*{mn},Z*{mn}\} and predicts:
• Objectives (\epsilon*{\max}, \tilde L*{\nabla B}, A) and constraints (QI residual, \Delta, W*\text{MHD}, \chi*{\nabla r}, \iota, etc.).
• A convergence classifier (VMEC high‑fidelity success/failure).
Use dataset metrics for supervision; fine‑tune with your own queried points (active learning). ￼
• Acquisition. For problems 1–2 use constrained BO (expected improvement under feasibility); for problem 3 use qEHVI on feasible regions, with trust‑region constraints in coefficient space to stay in the data manifold.
• Multi‑fidelity. Couple low‑fidelity VMEC++ as an inner check; promote to high fidelity only when: (i) surrogate’s uncertainty is low, (ii) predicted feasible, and (iii) predicted improvement is large. ￼
• Calibration. Calibrate uncertainties (e.g., ensembles or SWAG) so the feasibility classifier is conservative—zero scores are costly.

Track 2 — Learned generative priors with hard or explicit physics (precision)

Goal: Generate new feasible (or near‑feasible) boundaries with high coil‑simplicity at controlled A, then locally optimize.

Two complementary approaches from your PDFs: 1. PCFM (Physics‑Constrained Flow Matching) — hard constraint enforcement at inference.
Train a flow‑matching model in the Fourier‑coefficient space (optionally PCA‑compressed), then apply PCFM at sampling time to project intermediate samples onto constraint manifolds (e.g., QI residual below a threshold, bounds on \epsilon\_\text{max},\Delta,A,\iota). No retraining of the generative model is needed to enforce constraints, and PCFM is designed to satisfy nonlinear constraints exactly at the final state. Use VMEC++ metrics or cheap surrogates to define residuals and a differentiable projection. ￼ 2. PBFM (Physics‑Based Flow Matching) — embed residuals into the training loss.
Train the generator with a conflict‑free combination of flow‑matching and physics residual gradients; they report up to 8× lower physics residuals vs vanilla FM in PDE benchmarks. Here the residuals can be built from your surrogate versions of QI/MHD/turbulence metrics, with optional stochastic sampling for better coverage.

Practical recipe:
• Latent space: Start with PCA on \{R*{mn},Z*{mn}\} (as the paper also does for feasibility modeling/priors) and train FM in the low‑dimensional space; decode to coefficients. ￼
• Constraints in PCFM/PBFM:
• Equality/inequality residuals: QI residual \rightarrow r*\text{QI}(\Theta); mirror ratio, elongation, aspect bounds as ReLU residuals; optional VMEC‑convergence classifier penalty.
• For problem 3, add W*\text{MHD}\ge 0, \chi\_{\nabla r}\le c^\* residuals.
• Use as a proposal distribution: Sample batches of near‑feasible designs, filter with the fast surrogate, and polish with ALM+NGOpt in a small trust region.

Why this helps: the single‑objective scoring is binary on feasibility; sampling directly in (or near) the feasible manifold dramatically increases hit rate before expensive VMEC scoring. PCFM/PBFM are purpose‑built for that kind of constraint‑dominated generation.

Track 3 — Multi‑objective Pareto construction with physics priors (problem 3)

Goal: Build a dense Pareto front between A and \tilde L*{\nabla B} under constraints.
• Seeding: Use the PCFM/PBFM generator to seed feasible points across a range of A, then qEHVI refine (surrogate‑guided).
• Scalarizations on‑the‑fly: Randomize weights between A and \tilde L*{\nabla B} and solve a family of single‑objective subproblems (as in the paper), but with (i) better seeds and (ii) adaptive bounds on A to fill gaps. ￼
• Front regularity & repair: If VMEC‑high‑fidelity fails at a promising point, repair with local ALM steps and/or small coefficient smoothing (penalize high‑order modes) to improve convergence.

⸻

4. Engineering details that will save time
   • Boundary parameterization. Stick to stellarator‑symmetric Fourier modes (as required for scoring). Normalize each coefficient by a data‑driven scale; sort/group by (m,n) to exploit local structure in the surrogate. ￼
   • Preconditioning. Follow the paper’s diagonal scaling for optimization variables; decay scales with mode index; log‑transform the QI constraint for ALM stability. ￼
   • VMEC++ orchestration.
   • Low‑fidelity VMEC++ when exploring; high‑fidelity only for top‑K candidates.
   • Record force balance residuals and mesh quality; treat non‑convergence as a hard infeasibility in single‑objective problems. ￼
   • “Coil realism” sanity checks. The benchmark uses \tilde L\_{\nabla B} as a coil‑simplicity proxy; occasionally verify with a REGCOIL/NESCOIL pass to check coil curvature and clearance trends (off‑benchmark but useful for plausibility). ￼
   • Feasibility modeling. As in the paper’s generative appendix: use PCA + classifier (e.g., RF) to carve out a soft feasible region and GMM/MCMC to sample within it—good as a baseline proposal before the PCFM/PBFM work. ￼

⸻

5. Concrete, staged execution

Milestone A — One robust pipeline 1. Wire up constellaration evaluation locally (or Docker). Verify JSON schema round‑trip and metrics parity with the HF space. ￼ 2. Reproduce ALM+NGOpt for each problem with multi‑start, logging VMEC fidelity and constraint statuses; submit one baseline+ run per problem. ￼

Milestone B — Efficiency & feasibility 3) Train the surrogate stack (objectives, constraints, convergence). Add qEI/qEHVI with feasibility probabilities. 4) Add multi‑fidelity gating: LF‑VMEC\rightarrowHF‑VMEC promotion rules.

Milestone C — Generative priors 5) Train an FM generator in PCA‑space over feasible-ish points. 6) Implement PCFM (inference‑time hard projection) using surrogate residuals; compare with PBFM (training‑time residuals). Use the generator as a proposal engine; polish with ALM in a radius.

Milestone D — Pareto front (problem 3) 7) Combine generator seeds + qEHVI to fill the hypervolume faster; maintain a diversity term (k‑center in objective space) to avoid clustering.

⸻

6. Risks & mitigations
   • VMEC non‑convergence near sharp geometries → budget guardrails, coefficient smoothing, and repair steps; predict convergence with a classifier. ￼
   • Constraint “knife‑edge.” Single‑objective feasibility gate is unforgiving → PCFM projection + conservative surrogate uncertainties reduce 0‑score submissions. ￼
   • Distribution shift. Generator might propose out‑of‑manifold shapes → trust‑region in latent & coefficient space; rejection sampling via feasibility classifier (and LF‑VMEC).

⸻

7. Why this can beat the paper baselines
   • The paper’s baselines already note that ALM‑NGOpt is the only one that found feasible solutions across problems, but it used hours to tens of hours on many cores and still produced sparse fronts for the multi‑objective case. By adding:
   (i) surrogate‑aided acquisition,
   (ii) hard/explicit physics in generation (PCFM/PBFM), and
   (iii) multi‑fidelity gating,
   you improve both feasibility rate and oracle efficiency while discovering more diverse high‑quality designs. ￼

⸻

8. Notes & references used
   • Challenge overview, problems & scoring, dataset content, code and baselines: ￼ ￼ ￼
   • VMEC++ background & Python‑first interface: ￼
   • Flow‑matching with hard constraints at sampling (PCFM): ￼
   • Flow‑matching with physics residuals in training (PBFM): ￼
   • Lectures: Fourier boundary parameterization, quasisymmetry/QI context, coil proxies & MHD/turbulence optimization:

   1. Multi‑role agent workflow (ASI‑ARCH)

What the paper does
• Role separation with explicit gates. A Researcher → Inspector → Engineer → Analyzer loop proposes designs, checks novelty/sanity, runs and debugs code in a real environment, and then mines insights to steer the next round. The Inspector stage uses retrieval over prior work to test novelty, plus code‑level sanity checks (e.g., complexity and masking rules) and feeds failures back for rewrite. The Engineer runs training in an interactive coding sandbox and forces self‑revision from error logs; it also has automated QA to terminate unpromising runs early. The Analyzer keeps a “cognition” knowledge base distilled from ~100 papers and performs experiment‑aware analysis to guide the next designs. ￼
• Exploration‑then‑verification. Start with fast, low‑fidelity runs to triage many candidates, then scale only the promising ones for expensive, high‑fidelity evaluation. ￼
• Parallel search with shared memory. Many searches run in parallel against a shared experiment database so agents share what’s been tried and learned. ￼

Why this helps us
• Maps cleanly to fusion optimization:
Researcher proposes geometry/coils/solver settings; Inspector rejects obviously unsound proposals (e.g., poor complexity/feasibility heuristics) and enforces “physics sanity” gates; Engineer runs VMEC/field‑line/transport calls and must fix its own failures (e.g., non‑convergence); Analyzer learns which coefficients/constraints move the needle and writes back rules of thumb for the next round. The exact “self‑revision from logs” and auto‑termination are especially valuable when simulations fail or go off the rails. ￼

⸻

2. Targeted refinement & self‑ensembling (MLE‑STAR)

What the paper does
• RAG to form an initial solution, then component‑targeted exploration instead of rewriting whole pipelines each step. The agent runs ablation studies to discover which block matters most, then iterates there (e.g., feature engineering vs. model selection vs. ensembling). ￼
• Agent‑proposed ensembling. Rather than “best‑of‑n”, the agent merges multiple candidate solutions and refines the ensembling strategy itself. ￼
• LLM‑as‑judge to police contamination/novelty of the final solution relative to public write‑ups. ￼
• Demonstrated gains on MLE‑bench Kaggle tasks with minimal human input by using the above loop. ￼

Why this helps us
• In fusion design, “component‑targeted” translates to focused edits (e.g., tweak select boundary/coil coefficients, solver tolerances, or objective weights) guided by quick ablations rather than global rewrites every iteration. The self‑ensembling idea generalizes to portfolioed designs (keep several diverse geometries and blend/advance the best features) rather than single‑track search. ￼

⸻

3. Concrete practices we can adopt for Constellaration‑style agents
   • Gatekeeping before expensive runs (Inspector).
   • Novelty/redundancy check: retrieve similar historical designs or results, and reject near‑duplicates. ￼
   • Sanity checks: cheap structural/physical heuristics before VMEC/transport (e.g., bounds on curvature/coil spacing or coarse QS/well proxies). ASI‑ARCH’s code‑sanity and complexity checks are a blueprint for fast “go/no‑go” rules. ￼
   • Robust failure handling (Engineer).
   • Capture simulator error logs (non‑convergence, poor conditioning) and require the agent to propose fixes (resolution, preconditioning, step sizes, constraint relaxations). This exact “self‑revision from logs” mechanism is spelled out in ASI‑ARCH. ￼
   • Two‑stage evaluation.
   • Stage 1: low‑resolution equilibria, truncated mode sets, coarse coil discretization, or surrogate metrics.
   • Stage 2: high‑fidelity equilibria/transport only for shortlisted designs. That is the paper’s exploration‑then‑verification doctrine. ￼
   • Ablation‑guided edits (Analyzer → Researcher).
   • Run designed ablations over small sets of coefficients/constraints to learn sensitivity, then focus edits where the ablation shows leverage—mirroring MLE‑STAR’s “find the impactful block, then iterate there.” ￼
   • Maintain a domain “cognition base”.
   • Summaries of key stellarator papers (QS/QA/QI heuristics, near‑axis rules, coil complexity trade‑offs) in a structured KB and retrieve them by scenario (e.g., “reduce ITG drive at fixed rotational transform”). This mirrors the cognition base built in ASI‑ARCH. ￼
   • Self‑ensembling of designs.
   • Keep a small Pareto set; allow the agent to compose a new candidate that merges good traits (e.g., combine boundary modes from one design with coil smoothness from another)—the same spirit as MLE‑STAR’s ensemble merge. ￼
   • Parallel workers + shared memory.
   • Run many short jobs; store all runs and analyses in a shared DB so agents can avoid repeated dead ends and reuse partial wins. ￼
   • Tooling to expose gradients when possible.
   • Where differentiable paths exist (e.g., via SIMSOPT, a flexible stellarator optimization framework), plug gradient/adjoint information into the Analyzer so the Researcher proposes directed changes, not just random mutations. (SIMSOPT is cited in the uploaded stellarator reference list.) ￼

⸻

4. Minimal agent spec you can build now (pulled straight from the papers)
   1. Researcher
      • Inputs: Analyzer summary, retrieved “cognitions”.
      • Output: Parameterized proposal + rationale.
      • Guardrail: require a motivation and link to retrieved precedents (ASI‑ARCH). ￼
   2. Inspector
      • Novelty check via embedding retrieval against prior runs and literature; code/physics sanity gates; return actionable diffs on failure (ASI‑ARCH). ￼
   3. Engineer
      • Run simulation suite; on error, must fix using logs; cut off low‑promise runs early (ASI‑ARCH). ￼
   4. Analyzer
      • Archive results; run ablation‑style analyses (small controlled perturbations) to find leverage; update cognition base; recommend the next component to edit (MLE‑STAR + ASI‑ARCH).
   5. Portfolio/Ensembler
      • Periodically combine top designs into a synthesized candidate, not just vote or pick best‑of‑n (MLE‑STAR). ￼
   6. Scheduler
      • Implement exploration‑then‑verification and parallel batched workers with a shared experiment store (ASI‑ARCH). ￼

⸻

5. Why the non‑agent papers still matter here
   • While PCFM/PBFM are not agent papers, their notion of physics‑constrained objectives and feasible‑set enforcement gives us Inspector and Engineer checklists (residuals, boundary constraints, PDE consistency) to use as lightweight pre‑filters or training‑time penalties inside the loop. (Use them as “what to check,” not “how to act.”)
   • For optimization plumbing, the SIMSOPT reference confirms we can expose gradients/adjoints to the Analyzer when available and thereby steer proposals more efficiently than blind search. ￼

⸻

Bottom line
• From ASI‑ARCH: adopt role separation, self‑revision from logs, early termination, a knowledge base, parallel search, and two‑stage evaluation.
• From MLE‑STAR: use RAG to seed solutions, ablation‑guided targeted edits, and agent‑driven ensembling, plus an LLM‑as‑judge when we need qualitative/novelty checks.

Below is a full, implementation‑ready plan that a coding agent can execute to compete in (and aim to lead) the ConStellaration challenge. It combines (i) what the challenge requires and provides, and (ii) what we can leverage from recent physics‑constrained flow matching methods to generate, correct, and refine design candidates under hard constraints.

⸻

0. What the challenge actually asks you to do (succinctly)
   • Three benchmark problems (in rising difficulty): 1. Geometric – minimize max elongation at fixed aspect ratio, edge rotational transform per field period, and average triangularity. 2. Simple‑to‑build QI – minimize the magnetic‑field gradient scale length on the boundary (a coil‑simplicity proxy) subject to precise quasi‑isodynamicity (QI) and guard-rails on elongation/mirror ratio. 3. Multi‑objective, MHD‑stable QI – map the coil‑simplicity vs compactness trade‑off (hypervolume score), subject to QI, vacuum magnetic well, and a turbulence proxy constraint (flux‑surface compression in regions of bad curvature) at ρ=0.7. All evaluations are done with a VMEC++‑based forward model from the organizers. ￼ ￼
   • What Proxima/HF provide
   – A public dataset of ~150–180k QI‑like boundaries with VMEC++ equilibria + metrics (two tables: default & vmecpp_wout, linked by plasma_config_id).
   – Reference code, evaluation scripts, and baselines (augmented‑Lagrangian + Nevergrad, SciPy optimizers).
   – A leaderboard; submissions are Fourier‑represented boundaries; the scoring maps single‑objective results into [0,1] with feasibility gating; multi‑objective uses HV over feasible points. ￼ ￼
   • Tools you’ll use
   – Open‑source VMEC++ with modern C++ core and a clean Python layer, including helpful features such as hot‑restart; Proxima’s repo constellaration exposes the forward model + metrics/scorers. ￼ ￼

⸻

1. Big‑picture strategy

Two‑track loop with cross‑checks: 1. Fast generative proposals in the space of Fourier boundary coefficients using Flow‑Matching models trained on the dataset. We bake physics into the generator with PBFM during training (physics/algebraic residuals don’t fight the data objective) and apply PCFM during inference to project samples to the constraint manifold (hard constraints). 2. Robust forward‑model refinement with VMEC++ + Augmented‑Lagrangian Nevergrad (ALM–NGOpt), seeded by the best generative proposals. This achieves final feasibility and polishing against the true (organizer) metrics. ￼

This combination is designed to (a) generate many near‑feasible candidates quickly, (b) enforce the non‑negotiable constraints (QI residual, well, turbulence proxy, etc.), and (c) refine to SOTA objective values before submission.

⸻

2. System architecture a coding agent can implement

proj/
├─ env/
│ ├─ Dockerfile # pin Python 3.10, libnetcdf-dev, vmecpp>=0.4.6, constellaration
│ └─ conda.yml
├─ data/
│ ├─ hf_cache/ # Hugging Face dataset cache
│ └─ parquet/ # optional local mirrors
├─ conste/ # main Python package
│ ├─ io/ # dataset loaders, to/from VmecppWOut, exporter to VMEC2000
│ ├─ physics/ # metrics wrappers, constraint functions, scorers
│ ├─ fm/ # flow-matching models (PBFM training; PCFM inference)
│ ├─ surrogate/ # multi-output regressors for metrics, Jacobians (autodiff)
│ ├─ opt/ # ALM + Nevergrad loops; trust region; schedule
│ ├─ vmec/ # runners: coarse→fine; hot-restart; caching; error handling
│ ├─ agents/ # orchestration (DataOps, Train, Constraint, Runner, Submit)
│ ├─ eval/ # local eval parity tests; HV calculator
│ └─ viz/ # plots of boundaries, flux surfaces, Pareto fronts
└─ scripts/ # cli-entry points

Agents (lightweight components):
• DataOpsAgent – streams Hugging Face dataset, extracts columns boundary._, metrics._, vmecpp_wout.json, handles NFP slices, joins tables by plasma_config_id. ￼
• SurrogateAgent – trains multi‑output regressors (MLP/Transformer over Fourier series) to predict organizer metrics (objective + constraints) with uncertainty; exports autodiff models for Jacobians \partial h/\partial \theta.
• FMTrainAgent – implements PBFM training (flow‑matching loss + physics/algebraic residual loss, with temporal unrolling) over boundary vectors; supports deterministic & stochastic samplers. ￼
• FMGuideAgent – PCFM projection at inference: performs Gauss–Newton / Schur steps to project generated boundaries onto nonlinear constraint manifolds using surrogate Jacobians; chains multiple constraints safely. ￼
• VmecRunnerAgent – executes VMEC++ forward model w/ coarse→fine settings, hot‑restart, timeouts/retries, and result caching; emits full metrics via Proxima’s scorer. ￼ ￼
• RefineAgent – ALM–NGOpt outer loop with trust‑region proximal step (as in baselines), seeded by FM samples; escalates fidelity; stops when normalized constraint violation ≤ ε. ￼
• SubmitAgent – packages Fourier series into submission artifacts; runs local parity check against evaluation code; submits to HF leaderboard. ￼

⸻

3. Environment & reproducibility
   • OS/Deps: Ubuntu 22.04/24.04; libnetcdf-dev; Python 3.10 (as in repo tests); pin vmecpp ≥ 0.4.6 to avoid known Mac/attr mismatches reported by users; or use the provided Dockerfile. ￼ ￼
   • Install: pip install constellaration or clone → pip install . (repo exposes forward model + scoring). ￼
   • Dataset: datasets.load_dataset("proxima-fusion/constellaration", split="train") with streaming or local parquet; two parts (default, vmecpp_wout) joined via plasma_config_id. ￼
   • Version control: lock repo & dependency SHAs; export conda env; run nightly parity tests against organizers’ scorer (on a small batch) to catch drift.

⸻

4. Data representation & feature engineering
   • Input boundary representation: truncated Fourier series in cylindrical coordinates (DESC/VMEC form) with stellarator symmetry; consider up to |m|,|n|≤4 per the dataset generation and baselines. Normalize per‑mode amplitudes; include NFP as a categorical/one‑hot or embedded scalar. ￼
   • Targets (per problem):
   – P1 objective: max elongation \epsilon*{\max}; constraints: fixed A, \tilde\iota, \bar\delta (average triangularity).
   – P2 objective: normalized e*{L\nabla B}; constraints: QI residual small + bounds on \epsilon*{\max} and Δ; details & normalization follow Table 2 in the paper. ￼
   – P3 objectives: \langle e*{L\nabla B}\rangle vs compactness (A) Pareto; constraints: QI, vacuum magnetic well W*{MHD}>0, and turbulence proxy \langle\chi*{\nabla r}\rangle at \rho=0.7. ￼

The challenge evaluation maps single‑objective solutions into [0,1] with a linear rescale on the objective and feasibility gating; the multi‑objective score is hypervolume on feasible points. Implement local replicas of both. ￼

⸻

5. Surrogate models (metrics + Jacobians)

Goal: fast, differentiable approximations of the organizer metrics and constraints to enable (i) PBFM training signals, (ii) PCFM projection steps, (iii) efficient candidate screening.
• Model class:
– Start with MLP over the concatenated Fourier coefficients (grouped by (m,n)) + NFP embedding; train heteroscedastic outputs to quantify per‑metric uncertainty.
– Add a small Transformer encoder if correlations across (m,n) matter.
• Targets: predict \epsilon*{\max}, A, \tilde\iota, \bar\delta, e*{L\nabla B}, \text{QI residual}, W*{MHD}, \langle\chi*{\nabla r}\rangle, \Delta. (All are functions of boundary in the dataset.) ￼
• Training data: split by NFP and stratify across metric ranges to avoid covariate collapse; use the metrics.\* columns from the dataset’s default table. ￼
• Calibration: conformal prediction/temperature scaling; reject designs when surrogate uncertainty is high (trust‑region filter).
• Jacobians: implement with autograd (JAX/PyTorch). Export batched J = ∂h/∂θ for each constraint component h_i.

⸻

6. Generative design with Flow Matching

6.1 PBFM training (physics‑aware generator)

Train a Flow‑Matching model (deterministic and stochastic variants) to model the distribution of good boundaries. Add physics/algebraic residuals so that the generative flow aligns with feasible regions without fighting the data objective (temporal unrolling at training). This is PBFM: joint loss L=L*{FM}+L_R with no weight‑tuning conflict, and an effective stochastic sampler option for diverse, high‑fidelity samples. ￼
• What are “residuals” here? We use our surrogates to build differentiable residuals:
– P1: h=[A(\theta)-A^,\ \tilde\iota(\theta)-\tilde\iota^,\ \bar\delta(\theta)-\bar\delta^*].
– P2: h=[\text{QI}(\theta),\ \epsilon*{\max}(\theta)-\epsilon^{\max}{ub},\ \Delta(\theta)-\Delta{ub}].
– P3: h=[\text{QI}(\theta),\ -W_{MHD}(\theta),\ \langle\chi_{\nabla r}\rangle(\theta)-\chi^{\max}_{ub}].
All functions come from the surrogate (to keep training differentiable and fast).
• Conditioning: for P1/P2, condition on the fixed problem parameters (A^,\tilde\iota^,\bar\delta^\*,NFP) or the feasible ranges. For P3, sample a grid of aspect‑ratio targets and learn to produce diverse feasible solutions along the trade‑off.
• Unrolling: compute residuals at the temporally unrolled end‑state as in PBFM for better accuracy without extra inference cost. ￼

6.2 PCFM inference (hard‑constraint projection)

At sampling time, every generated boundary \theta is projected towards the constraint manifold with a Gauss–Newton step:

\theta \leftarrow \theta - J^\top (JJ^\top)^{-1} h(\theta),

iterated until \|h\|\le \epsilon. This is the PCFM projection on the linearized manifold; it generalizes and strictly enforces nonlinear equality constraints (and can chain several of them). We implement the Schur‑complement batched solve with autodiff Jacobians from the surrogate. ￼

Notes for the agent:
• Stack constraints (e.g., QI, then well, then turbulence proxy) – PCFM tolerates chaining; monitor conditioning of JJ^\top. ￼
• For inequality constraints g(\theta)\le 0, project onto h(\theta)=\max(0,g(\theta)) or use a slack/kKT‑like clipping before projection.
• If constraints are simple/affine (e.g., fixing a few Fourier modes), PCFM reduces to an exact orthogonal projection (recovers ECI as a special case). ￼

⸻

7. VMEC++ forward evaluation and refinement
   • Coarse→fine schedule: 1. Coarse run (fast): fewer flux surfaces & relaxed force tolerance to screen candidates. 2. Fine run (scoring fidelity): settings aligned with the organizer scorer.
   The repo’s hot‑restart drastically speeds repeated solves around nearby boundaries. ￼
   • Caching keyed on Fourier vectors; retry with slightly perturbed seeds if convergence fails (common in stellarator optimization).
   • Refinement: seed ALM–NGOpt from the top K PCFM‑projected samples (per problem). Use the non‑Euclidean proximal ALM with normalized constraint tracking (as in the baselines), shrinking the trust region as the duals settle. Keep the coarse forward model inside the inner loop and only promote to fine at convergence. ￼

⸻

8. Problem‑specific playbooks

P1 – Geometric
• Condition FM on (A^,\tilde\iota^,\bar\delta^\*,NFP);
• PCFM constraints: enforce [A,A\_target], [\tilde\iota,\tilde\iota\_target], [\bar\delta,\bar\delta\_target];
• Objective: minimize \epsilon\_{\max} → pick top candidates by surrogate, then forward‑evaluate and refine.
• Score: apply organizer’s single‑objective mapper with feasibility gating. ￼

P2 – Simple‑to‑build QI
• Residuals in PBFM: drive QI residual→0, respect bounds on \epsilon*{\max} & Δ, while encouraging low e*{L\nabla B}. Table 2 definitions and scalings from the paper. ￼
• PCFM: chain \text{QI}=0 then cap \epsilon*{\max}, \Delta.
• Selection: top on (low) e*{L\nabla B} under surrogate feasibility → forward model (fine) → ALM polish.

P3 – Multi‑objective, MHD‑stable QI
• Train FM conditioned on aspect‑ratio bins; use stochastic FM sampler to reliably populate the feasible frontier (PBFM showed stochastic sampling helps on complex distributions). ￼
• PCFM constraints: \text{QI}=0, W*{MHD}>0, \langle\chi*{\nabla r}\rangle \le \chi^{\max}\_{ub}. The turbulence proxy and the well definition follow the dataset paper; the proxy is evaluated at \rho=0.7. ￼
• Front building: run ALM from samples spanning aspect‑ratio targets; compute HV locally (same reference point as organizers) and pick a diverse representative set to submit. ￼

⸻

9. Implementation details the agent should code

9.1 Constraint & metric API

# conste/physics/constraints.py

@dataclass
class ConstraintSpec:
name: str
kind: Literal["eq","ineq"]
target: Optional[float] = None
ub: Optional[float] = None
lb: Optional[float] = None

class ConstraintFn(Protocol):
def value(theta: Tensor) -> Tensor: ... # h(theta) (batch, m)
def jacobian(theta: Tensor) -> Tensor: ... # J(theta) (batch, m, d)

# backed by SurrogateAgent (autodiff), not VMEC++.

9.2 PCFM projection (batched, Gauss–Newton / Schur)

def pcfm*project(theta, constraint_fn, tol=1e-3, iters=10):
for * in range(iters):
h = constraint_fn.value(theta) # (B, m)
if (h.abs().max() < tol): break
J = constraint_fn.jacobian(theta) # (B, m, d) # Solve (JJ^T) λ = h ; δθ = -J^T λ
JJt = bmm(J, J.transpose(1,2)) # (B, m, m)
lam = solve(JJt, h) # batched linear solve
dtheta = -bmm(J.transpose(1,2), lam) # (B, d)
theta = theta + dtheta
return theta

This is exactly the projection step PCFM motivates (orthogonal projection onto the linearized constraint manifold; Schur solve), and is robust & batched. ￼

9.3 PBFM training loop skeleton

# conste/fm/pbfm_trainer.py

for batch in data:
theta0 ~ prior() # boundary prior in coeff space
theta1 ~ data_batch # real boundaries # Flow Matching loss (Rectified Flow / FM)
L_fm = fm_loss(v_theta, theta0, theta1, t) # Temporal unrolling to t=1-eps, compute residuals with surrogate:
theta_e = evolve_to_end(v_theta, theta0, t_grid)
h = residuals(theta_e, spec) # h(...) stacked per problem
L_R = residual_loss(h) # algebraic / physics residual
loss = L_fm + L_R # conflict-free combo per PBFM
loss.backward(); optimizer.step()

Why this? PBFM shows (i) joint residual + flow‑matching without loss‑weight tuning conflicts, (ii) temporal unrolling improves end‑state accuracy, (iii) stochastic sampling often yields better distributional match. ￼

9.4 Refinement (ALM–NGOpt)
• Implement proximal‑ALM with non‑Euclidean trust region (baseline method), normalized constraint violations, diagonal preconditioning on variables (Fourier magnitudes decay by mode order). Use Nevergrad’s NGOpt as inner‑solver; keep VMEC++ coarse inside loop; check feasibility with fine VMEC++ on convergence. ￼

⸻

10. Evaluation parity, submission, and CI
    • Local evaluation: re‑implement the organizer’s scoring logic (single‑objective re‑scaling with feasibility cutoff; multi‑objective HV). Validate on the baseline configs and ensure scores match (±1e‑6). ￼
    • Submission packaging: format Fourier series exactly as specified on the dataset card/paper; include NFP & symmetry flags. Validate with the repo’s utilities (to_vmec2000_wout_file etc.) for sanity plots. ￼
    • CI checks:
    – Quick PCFM projection unit test on toy affine constraints (compares to closed‑form projection) – recovers ECI case. ￼
    – Surrogate Jacobian gradients finite‑difference check on a random batch.
    – VMEC++ mini‑batch run (3 candidates) both coarse/fine, ensure successful convergence and metric extraction.

⸻

11. How we’ll get from zero → leaderboard

Milestone A — Baselines & data 1. Container + install (libnetcdf-dev, constellaration, vmecpp>=0.4.6). Run Boundary Explorer notebook to verify plots & metrics. ￼ 2. Recreate ALM–NGOpt for P1 (3‑period devices as in the paper) and confirm local score parity. ￼

Milestone B — Surrogates 3. Train multi‑output surrogates on metrics.\*. Report R^2/MAE per metric; calibrate uncertainty; add trust‑region gating.

Milestone C — Generative core 4. Train PBFM models per problem (NFP‑specific or conditioned). Add temporal unrolling; enable stochastic sampler variant. ￼ 5. Implement PCFM projection (batched Gauss–Newton) over the problem’s constraint set; chain constraints and test numerical stability. ￼

Milestone D — VMEC++ + refinement 6. Build coarse→fine VMEC++ runners with hot‑restart and caching. Screen 1k FM samples → keep top‑p by surrogate objectives + PCFM feasibility → run coarse VMEC++ → refine K best with ALM → verify with fine VMEC++. ￼ 7. For P3, sweep aspect‑ratio bins, collect feasible refined points, compute HV, pick a diverse subset along the front for submission. ￼

Milestone E — Submission & iteration 8. Package boundaries; run organizer scripts locally; submit to the HF leaderboard. Track regressions in CI. ￼

⸻

12. Risk management & mitigations
    • VMEC++ non‑convergence: fall back to hot‑restart from neighbor; slightly increase resolution; or relax then tighten force tolerances (coarse→fine); perturb boundary by small Gaussian on high‑order Fourier modes. ￼
    • Surrogate mismatch: enforce a trust region (distance in feature space to nearest training samples); gate PCFM steps when uncertainty is high; escalate to physical refinement earlier.
    • Constraint conflicts (especially P3): PCFM supports chaining but conflicting constraints will stall; detect by growth of \|(JJ^\top)^{-1}\| and switch to multi‑stage projection or soft‑barriers before hard projection. ￼
    • Distributional collapse: prefer stochastic FM sampling; add K‑centers diversity selection before refinement. ￼

⸻

13. Why this should work
    • The organizers’ baselines show ALM–NGOpt is the only method among the three that reliably achieves feasibility across P1 & P2; we keep it as the final polisher, but seed it with better initial guesses from our FM+PCFM pipeline. ￼
    • PBFM gives a principled way to embed physics/algebraic residuals into the training of the flow, with temporal unrolling and stochastic sampling advantages—exactly what we need to bias the generator near feasible regions without hurting generative fit. ￼
    • PCFM gives a zero‑shot, inference‑time mechanism to enforce hard, nonlinear constraints via projection steps—perfect to impose QI, magnetic well, and turbulence‑proxy bounds before we pay for VMEC++. ￼

⸻

14. Acceptance criteria (done = shippable)
    • Parity tests pass on organizer examples; local scores match within tolerance. ￼
    • P1: outperform the baseline \epsilon*{\max} at fixed (A,\tilde\iota,\bar\delta) on the leaderboard. ￼
    • P2: strictly feasible QI with lower e*{L\nabla B} than the baseline; robustness shown across several NFPs. ￼
    • P3: submit a diverse Pareto set with improved HV over the public baseline front. ￼
    • Ablations: (i) w/ and w/o PCFM projection; (ii) deterministic vs stochastic FM sampler; (iii) with/without temporal unrolling; (iv) ALM seeded by random vs FM seeds. ￼

⸻

15. Domain references the code should rely on while implementing
    • Challenge & problems (official definitions, scoring, and dataset structure). ￼ ￼ ￼
    • VMEC++ capabilities & practicalities (hot‑restart, Python integration). ￼
    • Physics‑constrained generation:
    – PCFM for inference‑time hard‑constraint projection (Gauss–Newton/Schur, batched). ￼
    – PBFM for training flow models with embedded residuals, with temporal unrolling and stochastic sampling benefits. ￼
    • Background on metrics & proxies (QI, turbulence proxy, vacuum well; Fourier parameterization).

⸻

Appendix: physics primers useful for the team
• QI/QA/QH classes; why QI is attractive and tricky; Fourier boundary parameterization; coil simplicity proxy e\_{L\nabla B}; stability & turbulence proxies. (Great concise treatment in Landreman’s lecture notes.)

⸻

constelx/
├── README.md
├── pyproject.toml # installs 'constelx' CLI
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml # Ubuntu CI incl. libnetcdf-dev, lint, tests
├── src/constelx/
│ ├── **init**.py
│ ├── cli.py # Typer-based CLI (see commands below)
│ ├── data/
│ │ └── dataset.py # HF dataset loading & parquet export
│ ├── physics/
│ │ ├── constel_api.py # thin wrapper around `constellaration`
│ │ ├── constraints.py # stubs for aspect ratio, smoothness, etc.
│ │ ├── pbfm.py # Physics-Based Flow Matching skeleton
│ │ └── pcfm.py # Physics-Constrained Flow Matching skeleton
│ ├── optim/
│ │ └── evolution.py # CMA-ES baseline (optional extra)
│ └── surrogate/
│ └── train.py # MLP baseline; hook for FNO/DiT
├── tests/
│ └── test_cli.py
├── docs/
│ └── ROADMAP.md # step-by-step build-out checklist
└── LICENSE (MIT)

Why this maps cleanly to the ConStellaration challenge
• Data: constelx data fetch streams the ConStellaration dataset from the Hub and materializes filtered subsets (e.g., NFP=3) into Parquet for quick experimentation. The dataset provides ~150–180k QI-like equilibria with plasma boundary coefficients and VMEC++ “WOut” equilibrium outputs; columns such as boundary.r_cos, boundary.z_sin, and a vmecpp_wout part are already accounted for in the loaders. ￼
• Evaluation: the physics/constel_api.py wrapper calls the constellaration package’s forward utilities and scoring helpers (the repo ships a forward model + scoring functions). This package requires the NetCDF system dependency (libnetcdf-dev) — CI and README call that out for Ubuntu. ￼
• VMEC++: the underlying equilibrium code is modernized, open-source VMEC++, with Python integration and hot‑restart features; the skeleton assumes you’ll use the Python-facing interfaces exposed by the constellaration package. ￼
• Tasks: The CLI structure covers the three benchmark directions (geometry optimization, simple‑to‑build QI, multi‑objective QI with MHD stability) by letting you swap objective functions / metrics and mix classical + ML methods. ￼

⸻

Quick start

# 1) Create env

python -m venv .venv && source .venv/bin/activate

# 2) System dependency for VMEC++ I/O (Ubuntu)

sudo apt-get update && sudo apt-get install -y libnetcdf-dev # required by `constellaration` [oai_citation:4‡GitHub](https://github.com/proximafusion/constellaration)

# 3) Install

pip install -e ".[dev]" # + add ",[evolution]" for CMA-ES, " ,[bo]" for BoTorch

# 4) Sanity checks

constelx --help
constelx data fetch --nfp 3 --limit 32
constelx eval forward --example

CLI commands (from src/constelx/cli.py):
• constelx data fetch [--nfp INT] [--limit N]
Streams Hugging Face dataset and saves an ML-friendly Parquet subset of boundary. / metrics. columns\*_. ￼
• constelx eval forward [--boundary-json FILE | --example]
Runs a boundary through the forward/evaluator wrapper (metrics scaffolded).
• constelx opt baseline --algo cma-es --steps 50
Simple CMA‑ES baseline over a tiny boundary subspace (optional cma extra).
• constelx surrogate train
Trains a small MLP on (boundary._) → (metrics.\*) as a placeholder surrogate.
• constelx agent run
Skeleton multi‑step propose → simulate → select → refine loop to plug your agents into.

⸻

What the skeleton implements (and what’s stubbed)

Data & evaluation
• Dataset access via datasets.load_dataset("proxima-fusion/constellaration"); loader keeps Fourier boundary and metrics columns for speed, but it’s easy to widen the slice. Example usage in the README mirrors the Hub’s dataset card. ￼
• Forward model wrapper (physics/constel_api.py) using constellaration helpers to:
• build a minimal VMEC++ “WOut” from a boundary and
• compute compactness/smoothness style geometry metrics (scaffolded in code; swap in the exact scoring functions you’ll target from the repo). ￼

Baselines and surrogates
• CMA‑ES baseline (optim/evolution.py) exposes a tiny two‑parameter helical perturbation and a toy score aggregator. It’s deliberately small so you can quickly verify end‑to‑end evaluation and swap in your true objective(s).
• MLP surrogate (surrogate/train.py) shows the full loop (boundary coeffs) → (one scalar metric). Replace with FNO/DiT later.

Physics-constrained ML hooks (PBFM / PCFM)
• PBFM training utilities (stub) in physics/pbfm.py: add a physics residual alongside flow‑matching and combine gradients with a conflict‑free update direction (ConFIG‑style) to avoid trading off distributional vs physics loss; includes temporal unrolling notes to improve the final prediction used for residuals and a stochastic sampler option at inference. ￼
• PCFM sampling utilities (stub) in physics/pcfm.py: guide sampling with Gauss–Newton projections onto the constraint manifold, a reverse update via OT displacement, an optional relaxed penalty correction, and a final hard projection to satisfy equality constraints exactly at the end of the trajectory. Use this to enforce mass/flux/boundary consistency on generated fields. ￼

These stubs are documented inline with TODOs and function names that match the papers’ core steps, so your agent can implement them incrementally.

⸻

How to extend toward the three challenge tracks

The challenge defines three optimization problems (geometry opt; simple‑to‑build QI; multi‑objective QI with MHD stability) and provides reference implementations plus a physics‑based evaluation stack. ￼

    1.	Geometry‑first
    •	Add constraints/metrics in physics/constraints.py (e.g., aspect ratio, triangularity, twist).
    •	Tie them into the CMA‑ES objective or BoTorch acquisition (see optim/).
    2.	Simple‑to‑build QI
    •	Use smoothness/coil‑simplicity proxies and QI metrics exposed by constellaration.
    •	Add a PCFM-driven generator to enforce hard smoothness/flux constraints at inference.  ￼
    3.	Multi‑objective, MHD‑stable QI
    •	Introduce a multi‑objective driver (Pareto archive) with stability metrics from the evaluator;
    •	Train a surrogate with PBFM so residuals (divergence / analytic relations) are reduced without sacrificing distributional fidelity.  ￼

⸻

Notes your agent will care about
• System dependency: on Ubuntu, install libnetcdf-dev (and equivalent on macOS) before using the forward model in constellaration. The official README shows the exact apt / brew commands and a Dockerfile; CI in this repo mirrors that. ￼
• Dataset schema: the dataset ships two parts linked by plasma_config_id (default with boundary + metrics and vmecpp_wout with the VMEC outputs). Columns like boundary.r_cos, boundary.z_sin, and JSON payloads for vmecpp_wout are documented on the dataset card; the loader and the surrogate examples rely on these names. ￼
• VMEC++: modern C++ rewrite with Python integration and hot‑restart for faster sweeps — useful if you later step outside the provided forward tools. ￼
• Repo with evaluator & notebooks: Proxima’s proximafusion/constellaration repo contains the forward model, scoring, notebooks, and an installable PyPI package (pip install constellaration). This starter repo imports it directly. ￼

⸻

Minimal first run (suggested sequence) 1. Fetch a slice

constelx data fetch --nfp 3 --limit 1000

    2.	Forward test on a synthetic boundary

constelx eval forward --example

    3.	Run a small CMA‑ES baseline

pip install -e ".[evolution]"
constelx opt baseline --steps 30

    4.	Train the quick MLP surrogate

constelx surrogate train

    5.	Start wiring your agent

constelx agent run

⸻

Where to plug in the research
• Implement PCFM’s shoot → Gauss–Newton project → OT reverse → relaxed correction → final hard projection as a reusable guided_sampler for your generative model’s trajectories (physics/pcfm.py). This is tailored for hard equality constraints and tested on PDEs with shocks/nonlinear dynamics. ￼
• Implement PBFM’s conflict‑free gradient update + temporal unrolling in your training loops (physics/pbfm.py) to reduce residuals up to order‑of‑magnitude while preserving distributional accuracy; add a stochastic sampler option at inference when distributions are complex. ￼

⸻

Great question. Short answer: use a hybrid setup—your Mac is excellent for prototyping, data wrangling, VMEC++ runs, and light ML; switch to a cloud GPU (A100/H100/L40S class) when you’re ready to train bigger models or sweep lots of candidates.

⸻

TL;DR
• Local (your Mac): perfect for the repo, dataset exploration, feature engineering, baseline optimizers, VMEC++ evaluation, small-batch ML (PyTorch MPS). VMEC++ is designed to run on laptops and has a Python API, so you can iterate quickly without round‑trips to the cloud. ￼
• Cloud GPU: use when you train medium/large generative models (FNO/DiT/Transformers), do multi-seed ablations, or run many evaluations in parallel. The ConStellaration dataset is large (≈182k–314k rows; ≥4.3 GB just in the “first 5 GB” viewer), and the challenge uses VMEC++-based physics metrics and a leaderboard—both benefit from scaling. ￼ ￼

⸻

Why hybrid fits this challenge

What the challenge provides/asks:
• An ~160k QI‑like stellarator boundary dataset, plus benchmark problems and baselines. Submissions are boundary shapes (truncated Fourier series) that are evaluated with a physics forward model (VMEC++) and ranked on a leaderboard. ￼
• The dataset card shows the split layout (e.g., default and vmecpp_wout parts), approximate row counts and on‑hub size, and example code to stream/process with datasets+Dask. ￼

What this means for compute:
• Prototyping loop (data filters, plotting, boundary transforms, small optimizers, unit tests) ⇒ fast enough on your Mac. VMEC++ explicitly states it “can run on a laptop” and exposes a clean Python API; that’s ideal for the “inner loop.” ￼
• Training bigger generative/surrogate models or doing large sweeps (hyper‑params, seeds, population‑based search) ⇒ cloud GPUs save days of wall‑clock and let you keep the laptop responsive.

Two recent papers we’ve folded into the plan also inform sizing:
• PCFM (Physics‑Constrained Flow Matching): hard constraints are enforced at inference with a batched Gauss–Newton/Schur update; per‑step cost scales with small constraint dimension and overall remains roughly linear in state size. In practice the relaxed penalty needs only a handful of gradient steps—good news for running PCFM guidance on a laptop during sampling/tuning. ￼
• PBFM (Physics‑Based Flow Matching): adds a residual term and temporal unrolling at training to reduce physical residuals (often dramatically) while not increasing inference cost. That extra training work is where a cloud GPU shines; once trained, inference stays light. ￼

⸻

What to do locally on your Mac (recommended) 1. Repo + environment
• Use Python 3.10–3.12, uv or venv.
• Install VMEC++ from wheels/source: pip install vmecpp (they document macOS steps). ￼
• PyTorch with MPS backend for Apple GPU. Set PYTORCH_ENABLE_MPS_FALLBACK=1. 2. Dataset handling
• Stream from the Hub with datasets (num_proc + filtering) and keep a thin “ML slice” parquet locally for fast iteration. The dataset card shows example filtering and an estimated 182k–314k rows; plan space accordingly. ￼ 3. Inner loop
• Implement the baselines and your boundary generators, plot cross‑sections/flux surfaces, and wire VMEC++ calls. VMEC++ hot‑restart is handy to accelerate nearby configurations and is available in the Python API. ￼
• Prototype PCFM inference guidance on small batches; it’s compute‑light at sample time. ￼ 4. Small models
• Start with compact FNO/U‑Net/DiT variants on MPS for correctness and interfaces, not performance. When shapes/IO/contracts are stable, move to cloud.

⸻

When to go cloud, and what to rent

Triggers to switch
• You’re training PBFM/DiT/FNO models with unrolling or large contexts. ￼
• You’re running big sweeps (optimizers, seeds, constraint mixes) or many VMEC++ evaluations concurrently.

Recommended SKUs
• Single‑GPU, strong default: NVIDIA A100 40–80 GB (AWS p4d / GCP A2 / Azure ND A100 / Lambda Cloud / Runpod).
• For largest models/unrolling: H100 80 GB (AWS p5 / GCP A3 / Azure ND H100).
• Cost‑savvy / mixed workloads: L40S 48 GB or L4 24 GB if memory fits and you lean more on CPU+I/O.
Use BF16/mixed precision & gradient checkpointing to fit more. (No CUDA on Apple GPUs; that’s why cloud NVIDIA matters.)

CPU‑heavy sweeps
If you need to spray VMEC++ over thousands of boundary candidates, it’s a CPU/OpenMP code—spin up 64–128 vCPU instances (e.g., AWS c7i/c7gn, GCP c3‑highcpu) and shard with Ray/Slurm. VMEC++ is built for this style of batch evaluation. ￼

⸻

A pragmatic workflow 1. Develop local
• Implement data module, boundary parameterizations, scoring wrappers, and CLI.
• Validate end‑to‑end on small slices: boundary → VMEC++ → metrics → leaderboard format. (Leaderboard is a HF Space; submissions are boundary Fourier coefficients.) ￼ 2. Containerize once
• Dockerfile with two targets: cpu (VMEC++ + CLI) and cuda (adds PyTorch‑CUDA). Publish to GHCR so the same image runs on any cloud. 3. Burst to cloud for training/sweeps
• Launch a single‑A100/H100 VM with your image; mount an object store bucket; use accelerate/PyTorch Lightning for training and Ray for sweep orchestration.
• Track runs with W&B/MLflow (remote tracking server). 4. Back to local for analysis
• Pull artifacts, run PCFM‑guided sampling locally to test hard‑constraint satisfaction and produce leaderboard submissions. ￼

⸻

Concrete recommendations for you
• Start local today and keep using the Mac for:
• All code, unit tests, plots, dataset curation;
• VMEC++‑based metrics while you iterate on designs;
• Small‑batch PCFM‑style constrained sampling checks. ￼
• Add a cloud GPU for:
• Any PBFM training (residual loss + unrolling) or larger FM/DiT/FNO models; it improves residuals substantially without affecting inference cost, but training adds work that benefits from an A100/H100. ￼
• Optionally add a CPU cluster node when you need to evaluate many candidates with VMEC++ in parallel.

This split gives you fast iteration locally, and real acceleration only when/where it matters—exactly aligned with how the ConStellaration challenge is set up (dataset at scale, physics‑based scoring via VMEC++, leaderboard submissions). ￼ ￼

If you’d like, I can add a cloud/ folder to the starter repo with:
• Dockerfile.cuda + Dockerfile.cpu,
• a minimal terraform or modal script to bring up an A100 VM,
• Ray/Slurm job templates for VMEC++ batch evaluation,
• and accelerate configs for single‑GPU training.

    Awesome — you’re at the fun part now: turning the skeleton into a working, end‑to‑end system the GPT‑5 coding agent can iterate on.

Below I’ll (1) answer “production‑grade vs bit‑by‑bit,” (2) give you ready‑to‑paste instructions for the coding agent (as a doc you can drop into the repo), and (3) list concrete next actions (issues + acceptance criteria) so you and the agent can start committing code immediately.

⸻

1. Production‑grade or bit‑by‑bit?

Do a “vertical slice first,” then harden.
Ship a thin, end‑to‑end pipeline that goes: propose boundary → evaluate with constellaration → score → persist artifacts. Keep it minimal (CMA‑ES baseline, small dataset shard). Once it’s running, iterate in layers: 1. E2E slice (baseline CMA‑ES on a small subset) 2. Quality & speed (parallelism, caching, resumable runs, solid logging) 3. Learning models (surrogates) 4. Physics‑hard constraints (PCFM/PBFM modules plugged into the generation/selection loop) 5. Repro + CI + packaging (containers, pinned env, pre‑commit, GitHub Actions)

This approach gets you useful results fast and de‑risks integration with the evaluator and data format.

⸻

2. Paste‑in instructions for the AI coding agent

Create a new file:

docs/AGENT_PROMPT.md

# constelx — Coding Agent Runbook

## Prime objective

Implement, test, and document an end‑to‑end optimizer for ConStellaration:
propose boundary -> evaluate via `constellaration` -> compute score -> select next proposals.
Target correctness, determinism, and incremental performance. Prefer small vertical slices that run in minutes.

## Non‑negotiables

- Python ≥ 3.10, type hints everywhere, pass `ruff` + `mypy`.
- Tests: `pytest -q` must pass locally, and in CI (GitHub Actions).
- No breaking changes to CLI without updating docs and tests.
- Reproducibility: log configs, seeds, package versions, git SHA.
- Keep each PR < ~400 LOC if possible; include docs and tests.

## Working agreements

- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Add minimal integration test for every new CLI subcommand.
- For long‑running commands, add `--dry-run` and `--limit` flags.
- Always write a minimal example in the module docstring and cross‑link from README.

## Target architecture (already stubbed by the starter)

- `constelx.data`: dataset fetch/filter utilities, simple CSV index.
- `constelx.eval`: thin wrappers around `constellaration` evaluator; one function per metric + `score()`.
- `constelx.opt`: optimizers (`cmaes`, `bo_torch` stub).
- `constelx.models`: baseline MLP + scaffolding for FNO/transformers.
- `constelx.agents`: propose→simulate→select loop; checkpointing & resumability.
- `constelx.cli`: `constelx [data|eval|opt|surrogate|agent] ...`

## Minimum end‑to‑end (Sprint 0)

**Goal**: `constelx agent run --nfp 3 --budget 50` finishes and writes a results table, a JSONL of proposals, and a best‐of run summary.

Required pieces:

1. `constelx.eval.boundary_to_vmec(boundary: dict) -> VmecBoundary`
2. `constelx.eval.forward(boundary) -> dict[str, float]`
3. `constelx.eval.score(metrics: dict[str, float]) -> float` (aggregate as per evaluator defaults)
4. `constelx.opt.cmaes.optimize(f: Callable, x0: np.ndarray, bounds, budget) -> best_x, history`
5. `constelx.agents.simple_agent.run(config)`: wraps (1)-(4), logs to `runs/<timestamp>/...`

### Constraints and safety

- Parameterize boundaries in a small fixed basis (start with circular + low‑order Fourier modes).
- Clip & validate before evaluation (bounds, aspect ratio sanity, coil clearance if available).
- Timeouts and retries around evaluator calls; graceful skip on NaNs.

## Physics‑hard constraints

- Implement a pluggable “correction hook” in `agents` with two strategies:
  - `eci_linear`: projection for linear constraints (IC/BC/value/region).
  - `pcfm`: step‑wise Gauss–Newton projection on a constraint residual `h(u)=0` with Jacobian via autograd (zero‑shot inference correction).
- Training‑time residual minimization hook `pbfm` for flow‑matching surrogates (conflict‑free gradient combination of FM loss and residual loss).

## Logging & artifacts (required)

- Write `runs/<ts>/config.yaml`, `runs/<ts>/proposals.jsonl`, `runs/<ts>/metrics.csv`, `runs/<ts>/best.json`.
- Include a `README.md` in each run folder with CLI used and env info.

## CLI behavior to implement

- `constelx data fetch --nfp 3 --limit 128`
- `constelx eval forward --boundary-file examples/boundary.json`
- `constelx eval score --metrics-file runs/<ts>/metrics.csv`
- `constelx opt cmaes --nfp 3 --budget 50 [--seed 0]`
- `constelx agent run --nfp 3 --budget 50 [--seed 0] [--resume PATH]`

## Testing checklist

- Unit: boundary validation, scoring math, CMA‑ES step.
- Integration: one agent run with `--budget 5 --limit 8` finishes < 60s and writes artifacts.
- Golden files: store a tiny fixtures boundary & metrics JSON for regression tests.

## Performance rules

- Vectorize evaluator calls when possible; otherwise process pool with a small `--max-workers`.
- Cache per‑boundary derived geometry (hashable key) in a local sqlite or diskcache.

## Definition of done (per task)

- Working code + tests + docstring example + one paragraph in README.
- CI green.
- No perf regressions on the small e2e test.

⸻

3. What to do right now (tasks for the agent)

Open these as GitHub issues (copy/paste titles + acceptance criteria). The agent can pick up in order.

Milestone “Sprint 0 — Vertical slice”

#1 Wire evaluator wrapper (constelx.eval)
• Deliver: boundary_to_vmec(), forward(), score(), and a --example path already shown in README.
• AC:
• constelx eval forward --example prints a metrics dict and returns exit code 0.
• score() returns a single float; docstring states how the aggregate is formed.
• Unit tests cover: NaN handling, missing metric key → clean error.

#2 Minimal boundary parameterization + validation
• Deliver: constelx/eval/boundary_param.py
• AC:
• sample_random(nfp, seed) produces a dict with bounded Fourier coefficients.
• validate(boundary) raises on out‑of‑bounds; tested.

#3 CMA‑ES baseline
• Deliver: constelx/opt/cmaes.py, CLI constelx opt cmaes.
• AC:
• --budget 20 run produces runs/<ts>/metrics.csv and best.json.
• Handles evaluator failures via retry/backoff and logs skips.
• Has a dry‑run mode.

#4 Simple agent loop
• Deliver: constelx/agents/simple_agent.py, CLI constelx agent run.
• AC:
• End‑to‑end with --budget 20 --nfp 3 completes and writes artifacts.
• Resumable via --resume runs/<ts>.
• Integration test (tests/test_e2e_small.py) runs with --budget 5 fast.

#5 Logging & artifacts
• Deliver: a tiny run‑folder README generator + env capture.
• AC:
• runs/<ts>/README.md includes command, package versions, git SHA.

#6 CI & quality gates
• Deliver: .github/workflows/ci.yml (lint, type‑check, tests), ruff, mypy, pre-commit.
• AC:
• CI green on PR.
• Pre‑commit blocks missing type hints.

Milestone “Sprint 1 — Physics constraints & speed”

#7 ECI linear projection hook
• Deliver: constelx/agents/corrections/eci_linear.py
• AC:
• Given linear constraints A u = b for a boundary descriptor, the projection returns a feasible corrected proposal.
• Unit test with synthetic A, b.

#8 PCFM correction hook (inference‑time)
• Deliver: constelx/agents/corrections/pcfm.py
• AC:
• Pluggable correct(boundary) that solves one Gauss–Newton projection step; Jacobian via jax or torch.autograd depending on the boundary representation.
• --pcfm-steps k applies k corrections before evaluation.
• Small regression test on a toy constraint.

#9 Parallel evaluation & caching
• Deliver: process‑pool evaluator + disk cache (diskcache).
• AC:
• --max-workers respected, cached metrics re‑used within a run.

Milestone “Sprint 2 — Surrogates & PBFM hooks”

#10 Surrogate trainer skeleton
• Deliver: constelx/models/mlp.py, constelx/surrogate/train.py, CLI constelx surrogate train|predict.
• AC:
• Overfits a micro‑dataset (e.g., 1k proposals) and predicts metrics with R² > 0.8 on toy target.
• Saves and loads a .pt or .pth file; unit test runs CPU‑only.

#11 PBFM training hook (conflict‑free loss)
• Deliver: constelx/models/pbfm.py with a conflict‑free combination of FM loss and residual; unrolling support.
• AC:
• Unit test checks gradients flow and residual decreases on a toy residual.

⸻

4. How you personally kick this off (today) 1. Create the agent doc
   Add docs/AGENT_PROMPT.md (above) and commit. 2. Open the GitHub issues
   Use the numbered list; add labels: sprint-0, backend, good-first-task, test-needed. 3. Run sanity checks locally

source .venv/bin/activate
pip install -e ".[dev,bo]"
constelx --help
constelx data fetch --nfp 3 --limit 32
constelx eval forward --example

    4.	Give the agent its first assignment (paste as the very first message to GPT‑5):

You are the coding agent for constelx. Start with Issue #1 (“Wire evaluator wrapper”).

- Implement constelx.eval.{boundary_to_vmec, forward, score}.
- Add unit tests covering NaN handling and scoring aggregation.
- Add CLI: `constelx eval forward --example`.
  Constraints: keep changes ≤ 300 LOC, add docstrings + type hints, and make CI pass.
  When done, open a PR with a short summary and a bullet list of tests added.

      5.	Review PRs with tight scope

  Merge when tests + CI pass; keep PRs small and frequent.

⸻

5. When to switch from “slice” to “production‑grade”?

As soon as Sprint 0 completes (E2E runs reliably on your Mac and in CI), flip to hardening:
• Resilience: timeouts, retries, failure budgets.
• Speed: parallel eval, caching, smaller representations, vectorized scoring.
• Repro: pinned requirements.txt/uv.lock, lockfile in the run folder, optional Docker image.
• Docs: quickstart, examples, API docstrings surfaced with pdoc or mkdocs.

⸻

6. A few practical notes for your Mac
   • System deps: You already installed netcdf. If needed on macOS: brew install cmake pkg-config.
   • PyTorch (optional): Only if you train surrogates. Use the official Apple Silicon wheels from pytorch.org (MPS).
   • Fast local loops: Always run with --limit and --budget small while developing; add a tiny “dev dataset” in tests/fixtures/.

⸻

If you want, I can also provide:
• A ready‑to‑use .github/workflows/ci.yml,
• A minimal CONTRIBUTING.md,
• Example tests for the first two issues.

But you can already proceed now: add the agent prompt, open the issues, and send the agent the first task message.

Amazing—now that constelx_starter is live and your GPT‑5 coding agent is wired up, the fastest way to a production‑grade, actually‑result‑producing, self‑improving system is to take one thin end‑to‑end slice to production and then deepen each layer behind safety rails.

Below is a concrete, execution‑ready plan the agent can follow now. It turns the starter repo into a robust, continuously learning optimizer with physics‑aware generation and constraint enforcement.

⸻

What “production‑grade & self‑improving” means here

Definition of Done (DoD) 1. End‑to‑end autonomy: constelx agent run continuously proposes → evaluates (VMEC++/constellaration) → logs → learns → proposes again. 2. Safety rails: every stage has retries, timeouts, caching, and checksum’d artifacts; no silent failures. 3. Reproducibility: fixed seeds, pinned env, run manifests, and deterministic scoring; every result is one command away from re‑running. 4. Self‑improvement loop: active‑learning scheduler that (a) updates surrogate(s) from fresh evaluations, and (b) adapts the proposal distribution toward the Pareto frontier. 5. Physics‑aware generation: two levers:
• PCFM at inference to hard‑enforce equality constraints on generated candidates via tangent‑space projection + final Newton–Schur projection (Algorithm 1), guaranteeing constraint satisfaction at the sample’s end state. ￼
• PBFM at training to embed residual/algebraic physics into the flow‑matching objective with conflict‑free gradient updates and unrolled final‑state prediction—improving residual accuracy up to ~×8 without wrecking distributional fidelity. ￼ 6. Observability: structured logs, metrics dashboard, model cards for every surrogate release, and a CSV/Parquet leaderboard. 7. Ops: resumable jobs, concurrency limits, artifact store, and CI checks.

⸻

Immediate next steps (thin vertical slice → production)

Goal of this slice: one config, one command, measurable improvement run‑over‑run, and a file‑backed ledger of results.

0. Lock the environment
   • Freeze Python, constellaration, and VMEC++ deps in env/lockfiles/ (macOS/arm64 and Linux).
   • Add constelx doctor subcommand to verify NetCDF, VMEC++ and evaluator imports succeed.

1. Make the agent truly end‑to‑end

Wire the single‑run loop with hard failure handling and resumability.

Files to complete
• constelx/agent/loop.py
• Steps: propose_batch → evaluate_batch → postprocess → archive → learn → schedule_next
• Add: retry (exponential backoff), per‑candidate timeouts, and a --max-parallel semaphore.
• constelx/eval/runner.py
• Deterministic seeding, artifact paths, input hashing for evaluation caching.
• constelx/store/ledger.py
• Append‑only Parquet with schema: run_id, cand_id, params, metrics, status, walltime, git_sha, seed, artifacts_path.

CLI

constelx agent run --config configs/agent/vertical_slice.yaml --max-parallel 4

Acceptance checks
• If a VMEC++ job crashes or times out → status=failed with error trace; run continues.
• Re‑running the exact command does not repeat cached evaluations.

2. Ship two proposal modes (for diversity + stability)

   1. CMA‑ES baseline (constelx/opt/cmaes.py)

   • Box‑constrained center + sigma; penalty for inequality constraints (until we add hard projection).
   • Mutator: orthonormal basis perturbation on Fourier boundary coefficients.

   2. RND/Latin (constelx/opt/random.py)

   • Stratified sampling (Halton / LHS) for exploration.

Wire a mixture‑of‑proposers (constelx/agent/policy.py) with weights [0.6 CMA, 0.4 RND], annealed by recent hit‑rate (top‑k successes).

3. Scoring & selection gates
   • Implement constelx/eval/scoring.py with:
   • Primary score: weighted sum or Pareto dominance on the official metrics the evaluator returns.
   • Hard gates: reject candidates not VMEC‑feasible or violating obvious engineering bounds.
   • Leaderboard writer: artifacts/leaderboard.csv updated every N evaluations.

4. Surrogate v0 and self‑improving loop
   • Start lightweight: XGBoost/GP on metrics (no deep nets yet) in constelx/surrogate/tabular.py.
   • Active learning in constelx/agent/learn.py:
   • Fit surrogate from the ledger (only status=ok).
   • Acquisition = Expected Improvement under surrogate uncertainty; propose “surrogate‑guided” candidates (add as a third proposer at low weight).

After this slice, you’ll have a system that learns from its own runs and gets measurably better.

⸻

Add physics‑aware generation (the big win)

5. PCFM inference‑time hard constraints (generator side)

Use PCFM as a projection‑corrector inside the sampler that produces boundary parameter vectors. The agent takes any candidate x from a generator (model/CMA/random), then: 1. Forward shoot to a proposal state, 2. Gauss–Newton tangent projection onto the constraint manifold h(x)=0, 3. Relaxed correction (optional), 4. Final Newton–Schur projection to enforce h(x)=0 exactly, then commit the corrected x̂.

Implement:
• constelx/constraints/core.py — constraint interface: residual(x), jacobian(x) (batched).
• constelx/models/flows/pcfm.py — Algorithm 1 steps with a small batched solver; expose project(x).

What to encode as equality constraints first
• NFP symmetry patterns in Fourier modes.
• Periodicity/phase‑locking across poloidal–toroidal indices.
• Any evaluator‑provided exact equalities (e.g., boundary normalization or fixed origin).

PCFM guarantees exact satisfaction of h(x)=0 at the final state without retraining the generator, which is perfect for production—hard constraints, plug‑in, and zero‑shot. ￼

Where to call it
• In policy.py just before enqueueing a candidate to eval:

x = proposer.sample()
x_hat = pcfm.project(x) # hard-enforce equalities

6. PBFM training‑time physics (surrogate side)

When you move beyond tabular surrogates to neural flows/operators, train them with PBFM:
• Loss = Flow‑Matching + Physics Residual with conflict‑free gradient update (ConFIG), so both terms improve together instead of fighting.
• Implement g_update from Eq. (3) with the orthogonality operator (Appendix E in the paper).
• Temporal unrolling during training to refine the final noise‑free prediction before computing the residual, which yields much better residual gradients.
• Set σ_min thoughtfully; in physics settings, lower σ_min (even 0) avoids inflating residual floors.
• Optionally use a stochastic sampler at inference to improve distributional fidelity when complexity is high. ￼

Implement:
• constelx/surrogate/pbfm.py — trainer with unrolling, residual hooks, and conflict‑free updates.
• constelx/physics/residuals.py — cheap residuals/algebraic relations you can compute from evaluator outputs (e.g., analytic relations between boundary harmonics, conserved quantities that the evaluator exposes, or internal consistency checks).

PCFM (inference) + PBFM (training) gives you hard feasibility now and in‑distribution, physics‑consistent surrogates that keep getting better as the ledger grows. ￼ ￼

⸻

Production hardening (without slowing you down)

7. Caching, artifacts, and resumption
   • Input hash → output cache (NetCDF/JSON) in artifacts/evals/{hash}/.
   • Manifest file per run (run.yaml) with config, git SHA, environment, seeds.
   • constelx agent resume --run-id <id> picks up unfinished work.

8. Observability
   • logs/ with structured JSONL per stage; aggregate Prometheus‑style counters (evaluations/hour, success rate, cache hit‑rate, best score).
   • constelx report make --run-id ... to generate a Markdown/HTML report (leaderboards, traces, plots).

9. CI & tests
   • Unit tests: constraints (projection leaves h(x̂)≤1e‑8), caching (idempotent), scoring (Pareto), and agent (no deadlocks).
   • Golden test: a tiny config that evaluates 4 canned candidates and reproduces a saved leaderboard.

10. Configs for local and cluster
    • configs/agent/local.yaml (your Mac): max-parallel=4, shorter timeouts.
    • configs/agent/cluster.yaml: longer timeouts, larger batches; same codepath.

⸻

Example: production agent config (drop in as configs/agent/prod.yaml)

run:
name: "prod-main"
seed: 1337
out_dir: "artifacts/runs/prod-main"
max_parallel: 8
resume: true

proposers:

- name: cmaes
  weight: 0.5
  params:
  sigma0: 0.15
  popsize: 16
  bounds: "configs/bounds/nfp3_box.yaml"
- name: random_lhs
  weight: 0.3
  params:
  n: 16
  strategy: "halton"
- name: surrogate_guided
  weight: 0.2
  params:
  k: 12
  acquisition: "ei"

constraints:
hard_equalities: - type: "nfp_symmetry"
nfp: 3 - type: "phase_lock"
groups: - ["r_1_0", "z_1_0"]
pcfm:
enabled: true
steps: 10
relaxed_lambda: 0.0
tol: 1e-8

evaluation:
engine: "constellaration"
cache: true
timeout_s: 900
retries: 1

scoring:
primary: "weighted_sum"
weights:
physics_quality: 0.6
engineering_feas: 0.4
hard_gates: - "vmec_feasible" - "min_distance >= 0.03"

learning:
enabled: true
schedule: "every_n=64"
surrogate:
kind: "tabular_xgb" # later: "pbfm_flow"
pbfm:
enabled: false # flip to true when deep surrogate lands
unroll_steps: 4
sigma_min: 0.0
sampler: "stochastic"

⸻

How this becomes self‑improving 1. Every evaluation appends to the ledger. 2. The learner consumes the latest ledger slice, retrains a surrogate, validates, and version‑tags it. 3. The policy increases the weight of the surrogate‑guided proposer when its offline validation (on the last M batches) improves EI hit‑rate; otherwise, it decays. 4. PCFM keeps all generator outputs on‑manifold for equalities (prevents “garbage” work). ￼ 5. PBFM (once enabled) trains the flow surrogate with a physics residual and conflict‑free updates so it generalizes better and doesn’t trade physical correctness for likelihood. ￼

⸻

What to tell the coding agent (copy into your docs/AGENT_GUIDE.md)

Mission: “Given a config, run propose → pcfm_project → evaluate → learn → repeat, logging everything. Never crash; degrade gracefully and keep going.”

Order of work: 1. Implement caching + ledger writing. 2. Wire the agent loop with retries & timeouts. 3. Add CMA‑ES + Random proposers. 4. Add scoring & hard gates. 5. Add surrogate v0 + active learning. 6. Add PCFM projector and call it pre‑eval. 7. Add PBFM trainer (behind a feature flag). 8. Add reports & CI tests.

Non‑negotiable invariants:
• No candidate enters evaluation unless hard_equalities are satisfied (PCFM passes).
• Every artifact & metric is reproducible from the manifest.
• The agent must be resume‑safe.

⸻

TL;DR next command to run 1. Vertical slice (local):

constelx doctor
constelx agent run --config configs/agent/vertical_slice.yaml --max-parallel 4
constelx report make --run-id <printed-run-id>

    2.	Flip on PCFM projection once the loop is stable:

constelx agent run --config configs/agent/prod.yaml --max-parallel 4

    3.	Enable learning (surrogate) after the ledger has a few hundred successful evals; later switch surrogate kind to pbfm_flow and flip learning.surrogate.pbfm.enabled: true.

⸻
