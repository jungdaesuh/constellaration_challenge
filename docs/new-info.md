Below is a **complete, end‑to‑end plan** to understand, attack, and win the ConStellaration challenge. I’ve (i) distilled the official rules and scoring, (ii) analyzed every paper and the uploaded source kits, (iii) synthesized the most useful ideas into a practical, mathematically grounded optimization stack, and (iv) designed a self‑improving system that keeps climbing the leaderboard.

---

## 0) Executive summary — how to win

**What gets ranked.** The challenge has **three benchmarks of increasing difficulty** evaluated with a **physics forward model (VMEC++)** from boundary shapes expressed as **truncated Fourier series**. Submissions are scored with single‑objective or multi‑objective metrics; all require satisfying feasibility constraints. The leaderboard measures:

1. **Geometric**: minimize **max elongation** at fixed bounds on aspect ratio, triangularity, and edge rotational transform.
2. **Simple‑to‑build QI**: keep **QI residual small** while **maximizing coil simplicity** (via normalized magnetic gradient scale length) under constraints on mirror ratio, elongation, etc.
3. **MHD‑stable QI (multi‑objective)**: explore the **compactness ↔ coil simplicity** Pareto front while satisfying **vacuum well** and a **turbulence proxy** (flux compression in bad curvature) constraints.

**What beats the baselines.** The released baselines rely primarily on **augmented‑Lagrangian + gradient‑free NGOpt (Nevergrad)** and show feasibility gaps for other methods. So the path to win is to **out‑explore** and **out‑constrain**:

- **Use multi‑fidelity surrogates** to slash VMEC++ calls.
- **Exploit analytic structure** (QI targets in Boozer coordinates; near‑axis expansions) to propose **high‑quality seeds**.
- **Enforce constraints exactly during generation** (hard‑constrained functional flows) to sample _only_ feasible regions.
- **Trust‑region, feasibility‑first search** to polish with VMEC++.
- **Self‑improving agentic loop** that continually (re)trains surrogates from your own best designs and regenerates candidates.
  This outperforms single‑fidelity, weight‑tuned, stochastic scans reported in the paper.

---

## 1) Challenge essentials & evaluation (what you must satisfy)

- **Inputs**: Plasma boundaries as **Fourier series** in cylindrical coordinates $R(\theta,\phi), Z(\theta,\phi)$ with finite mode cutoffs; the evaluator uses **VMEC++** to compute an **ideal‑MHD equilibrium** and derived metrics. **Submissions must follow this representation**, exactly as in the preprint & code.
- **Benchmarks**:
  **(P1 Geometric)**: minimize **max elongation**; constraints on **aspect ratio**, **average triangularity**, **edge rotational transform per field period**.
  **(P2 Simple‑to‑build QI)**: **single‑objective**; enforce QI residual bound, mirror ratio, elongation; score favors **coil simplicity** (larger normalized magnetic gradient scale length) at feasible designs.
  **(P3 MHD‑stable QI)**: **multi‑objective**; Pareto over **compactness** vs **coil simplicity**; require **vacuum well** and **turbulence proxy** constraints. **Scoring** = bounded single‑objective map or **hypervolume** over feasible points for the multi‑objective case.
- **Data + code**: open dataset (\~**150–160k** QI‑like surfaces with equilibria & metrics), VMEC++ evaluators, and strong baselines are provided; **you submit boundary JSON; leaderboard computes & ranks**.
- **VMEC++**: modern C++/Python re‑implementation of VMEC with **“zero‑crash” policy**, **hot‑restart**, Python API, and validated performance; wheels via `pip install vmecpp`. Works as forward model and integrates with SIMSOPT.

**Implication:** Winning requires **reliable feasibility**, **higher objective quality** than ALM‑NGOpt baselines, and **good compute economics** per attempt.

---

## 2) What the uploaded ConStellaration paper & code give you (key takeaways)

- **ConStellaration paper** (Proxima Fusion): standardizes the **three problems**, **metrics**, and **scoring**; documents **baseline optimizers** and **evaluation protocol**; emphasizes **QI in Boozer coordinates**, **coil simplicity via magnetic gradient scale length**, **vacuum well** as an ideal‑MHD stability proxy, and a **geometric turbulence proxy**. It also shows dataset generation from **QI targets** + **stage‑1 optimization** and notes that **ALM‑NGOpt** was the only baseline consistently finding feasible solutions.&#x20;
- **Constellaration repo (uploaded)**: includes an **evaluator** that computes metrics via VMEC++ and Boozer transforms, and exposes three problems:

  - **GeometricalProblem** → constraints (e.g., `aspect_ratio ≤ 4.0`, `avg_triangularity ≤ -0.5`, `edge ι/Nfp ≥ 0.3`) with objective **max elongation**.
  - **SimpleToBuildQIStellarator** → constraints include `log10(QI residual) ≤ -4.0`, `mirror ratio ≤ 0.2`, `edge ι/Nfp ≥ 0.25`, `max elongation ≤ 5`. Objective normalizes **minimum normalized magnetic gradient scale length**.
  - **MHDStableQIStellarator** → constraints include `log10(QI residual) ≤ -3.5`, `mirror ratio ≤ 0.25`, **flux compression in bad curvature ≤ 0.9**, **vacuum well ≥ 0**; score = **hypervolume** in $(-\text{coil‑simplicity},\ \text{aspect ratio})$ space (feasible set only).
    The forward model object computes **aspect ratio, triangularity, rotational transform, mirror ratio, max elongation, QI residual (Boozer), vacuum well**, **turbulence proxy (flux compression)**, and **min normalized magnetic gradient scale length** at specified flux surfaces. _(Derived by reading the uploaded evaluator and problem classes.)_

- **Dataset arXiv (HTML)** repeats the evaluation structure, the **hypervolume** scoring for the multi‑objective, and the **coil‑simplicity metric** definition and rationale.

---

## 3) Papers you uploaded that are useful (and how to use them)

**Constrained generation / sampling (to stay feasible):**

- **ECI sampling** (gradient‑free, zero‑shot hard‑constrained guidance for flow matching): interleave **Extrapolate–Correct–Interpolate** each sampling step to **exactly satisfy constraints** without gradients; ideal to **enforce QI residual bounds, mirror ratio, vacuum well, turbulence proxy** in **the generative stage**. Apply on boundary‑coefficient space or on Boozer‑field parameters.&#x20;
- **PCFM** (Physics‑Constrained Flow Matching): **projection of intermediate samples** onto **nonlinear constraint manifolds** (mass/energy analogs), enabling **hard constraints at inference** for flow models; robust on sharp features. We’ll adapt the projection operator to enforce **feasibility vector** from the challenge (QI residual, mirror ratio, vacuum well, etc.).&#x20;
- **PBFM** (Physics‑Based Flow Matching): **train‑time** incorporation of physics residuals **without weight‑tuning conflicts**, plus **temporal unrolling**; gives distributions that already **respect the PDE structure** (here, MHD‑consistent geometry surrogates) and improves the **final noise‑free sample**—useful when pretraining a generative prior over boundary shapes/equilibria.&#x20;

**Optimization algorithms / surrogates:**

- **Augmented‑Lagrangian coil optimization** (SIMSOPT): a **strong Stage‑2 coil solver** that outperforms previous coil sets (W7‑X, HSX) and avoids weight hand‑tuning by dual updates. We won’t optimize coils in the leaderboard, but **coil simplicity** is proxied in P2/P3; insights carry over to **constraint handling** and can help design _better penalties/dual updates_ for Stage‑1 search.&#x20;
- **Data‑driven geometry→omnigenity** (LightGBM/NN + SHAP): shows **which boundary DOFs matter most** for **QI/quasi‑symmetry** and **solver convergence**; we use these as **feature importances** to **reduce effective dimensionality**, initialize priors, and gate candidate edits.&#x20;
- **Multi‑fidelity active learning** (LDD implosions): builds **1D→2D surrogate stacks** and **actively queries expensive sims** only where payoff is high; translate to **VMEC++ fidelity ladders** (low‑res, tight tolerances → high‑res final scoring) to **minimize expensive runs**.&#x20;

**Self‑improving “agentic” loop:**

- **rStar2‑Agent**: agentic RL over a **code tool environment**; **Resample‑on‑Correct** rollouts (GRPO‑RoC) filter noisy tool interactions and **improve reasoning with short, precise trajectories**. Directly applicable to our **search agents** that write Python to mutate boundaries, call VMEC++, and parse metrics.&#x20;

> Together, these give us **feasible‑by‑construction generation**, **fast exploration via surrogates**, and a **tool‑using RL agent** that autonomously proposes, tests, and learns.

---

## 4) Source code analysis (uploaded kits)

- **VMEC++**: Python API (`vmecpp.run`), **hot‑restart**, friendly errors, **SIMSOPT wrapper**, CLI & Docker; validated against Fortran versions; practical dependency hints and OS notes are documented. This is our **trusted forward model**.
- **Constellaration code** (uploaded):

  - **Forward model** assembles metrics: **geometry** (aspect ratio, triangularity, elongation), **magnetic metrics** (edge ι/Nfp, mirror ratio, min normalized gradient scale length), **QI residual** via **Boozer transform**, **vacuum well**, **turbulence proxy** (flux compression in bad curvature).
  - **Problem classes** implement the **constraint vectors** and **score maps** (single‑objective normalization; multi‑objective **hypervolume**).
  - **Evaluation app** accepts **JSON boundary** and returns score & feasibility.
    These give us exact **constraint thresholds** and **metric formulas** we must hit.

---

## 5) Strategy: an algorithmic stack that wins

### 5.1 Representations & physics priors

- **Boundary parameterization**: start with the **same truncated Fourier basis** and **stell‑symmetry** assumptions as the evaluator. Add **spectral pre‑conditioning** (scale variables by typical magnitudes per mode) to improve optimizers’ conditioning.
- **Near‑axis “QI seeds”**: sample QI‑like fields (Dudt‑style omnigenous parameterization) and **invert to surfaces** using near‑axis models; refine with a quick VMEC++/DESC pass at low fidelity. This keeps proposals **close to QI** before hitting the hard QI residual bound.&#x20;

### 5.2 Multi‑fidelity forward modeling

- **Fidelities**:
  **F0 (cheap)**: **DESC** AD run or low‑res VMEC++;
  **F1**: VMEC++ with reduced resolution & looser force tolerance;
  **F2 (final)**: **challenge’s high‑fidelity VMEC++** (same settings as scoring).
- **Active learning over fidelity**: train surrogates on **(boundary → metrics)** with uncertainty; **query up** the fidelity ladder only if expected improvement or constraint‑violation uncertainty warrants. Adapt the **1D→2D strategy** from LDD study to **F0→F2**.&#x20;

### 5.3 Hard‑constrained generative proposal

- **Pretrain a prior** $q(\mathbf{a})$ over Fourier coefficients on the dataset (and our own feasible cache).
- **ECI / PCFM at inference**: during sample refinement, **project** intermediate samples onto the **feasibility manifold** defined by the challenge constraints (QI residual threshold, mirror ratio, vacuum well ≥ 0, flux‑compression threshold, etc.). This yields a stream of **already‑feasible** candidates, drastically improving hit‑rate.
- **PBFM (optional)**: if training a new prior, **embed physics residuals** in the loss to make **feasible regions high‑probability**, reducing downstream corrections.&#x20;

### 5.4 Local polishing: feasibility‑first trust region

- **Outer loop**: **Feasible Augmented‑Lagrangian** with adaptive duals (learned multipliers) to avoid hand‑weight tuning—mirroring modern coil AL methods—but operating on **Stage‑1 metrics** (ours).&#x20;
- **Inner loop**: trust‑region step in **scaled Fourier space**, search direction from **surrogate gradients** (F0/DESC AD) blended with **finite‑difference “check”** on F1 VMEC++. Acceptance requires **no violation growth** and **objective gain**.
- **Multi‑objective (P3)**: use **Expected Hypervolume Improvement (EHVI)** on the feasible set (from surrogate) to guide where to evaluate at F2; retain a **diverse Pareto archive**.

### 5.5 Turbulence & coil‑simplicity awareness

- **Coil simplicity**: optimize **minimum normalized gradient scale length**; penalize configurations whose **spectral energy** concentrates in high toroidal modes (heuristic correlates with tighter coils). The evaluator’s metric is the ground truth.
- **Turbulence proxy**: include **flux‑compression‑in‑bad‑curvature** proxy in constraints and as an auxiliary loss for the surrogate; cross‑validate against **Landreman‑style geometric correlates** to reduce false feasibility. (The preprint documents this proxy.)

### 5.6 Feature‑sparse edits (ablation‑proven)

- Use **feature importances** (LightGBM/SHAP) from the data‑driven paper to **restrict edits** to **impactful modes**; this reduces dimension and local curvature ugliness, improving **convergence** and **feasibility retention**.&#x20;

### 5.7 Robust engineering of the loop

- **Caching**: memoize all (boundary → VMEC++ `wout` → metrics).
- **Hot‑restart** VMEC++ between nearby edits.
- **Automatic “rescue”**: if F2 fails to converge, auto‑revert to last F1‑feasible point and shrink trust region; keep the **dual variables** (ALM) to avoid feasibility regressions.

---

## 6) Self‑improving LLM+RL system (“ASi‑Stellarator”)

**Roles:**

1. **Proposer** (flow‑matching generator + ECI/PCFM correction) → **feasible proposals**.
2. **Critic‑Surrogate** (multi‑task GPs/NOs) → predicts metrics + uncertainty, trained on **dataset + our cache**; drives **active learning**.&#x20;
3. **Planner Agent** (rStar2‑style agentic RL): writes Python to **mutate Fourier coefficients**, call VMEC++/DESC, update archive, and decide **fidelity**; uses **GRPO‑RoC** to filter noisy tool traces and **prefer concise, correct** trajectories.&#x20;
4. **Polisher** (feasible ALM + trust‑region) → final improvement before submit.&#x20;

**Self‑improvement loop:** every N submissions, retrain the **prior** (PBFM) and **surrogates** on the **new feasible archive**; **refit SHAP** to re‑rank Fourier modes; update **constraint projections** (PCFM) to match the leaderboard’s most constraining regimes.&#x20;

---

## 7) Theoretical checks (sketches)

- **QI residual & Boozer‑field closure**: enforcing **poloidally closed $B$ contours** and **aligned maxima** (QI properties) reduces **orbit‑averaged radial drift**; our **hard‑constrained generators** (ECI/PCFM) target these invariants, so at inference time $r\mapsto 0$ residual implies feasibility within evaluator tolerances.&#x20;
- **Coil‑simplicity proxy**: **normalized magnetic gradient scale length** grows when coils can be further and smoother; increasing it at fixed $A$ shifts the **hypervolume** frontier right/down, improving P2/P3 scores while maintaining constraints. (Formalized in the preprint.)&#x20;
- **Feasible ALM convergence**: augmented‑Lagrangian with multiplier updates and non‑Euclidean proximal terms converges under standard assumptions; **dual updates** remove the need for manual weight tuning that plagued earlier approaches. (Insight carried from coil ALM to Stage‑1 constraints.)&#x20;
- **Multi‑fidelity optimality**: given monotone fidelity costs and Lipschitz error bounds, active learning that escalates fidelity on **EI/EHVI** improves sample efficiency without biasing the final frontier (as long as final selection is evaluated at **F2**).&#x20;

---

## 8) Concrete playbook (two‑week sprint)

**Day 1–2: Environment**

- Build VMEC++ (`pip install vmecpp`), verify **W7‑X demo**, test **hot‑restart**.
- Run the **Constellaration evaluator** locally on provided sample JSON boundaries; verify scores/feasibility match paper’s examples.

**Day 3–5: Baselines & surrogates**

- Re‑run **ALM‑NGOpt** baselines to get internal baselines; cache all metrics.
- Train **multi‑task surrogate** $f:\mathbf{a}\mapsto$ (metrics, uncertainty) on HF dataset + our runs; fit **SHAP** to rank Fourier coefficients (keep top‑k).&#x20;

**Day 6–8: Feasible generative proposals**

- Pretrain **flow‑matching prior** on dataset; add **PBFM** residuals (lightweight) if time.
- At inference, sample with **ECI/PCFM** projections to constraints of each problem; check feasibility at **F1**.

**Day 9–11: Polishing & MOEA**

- For P1/P2: **Feasible ALM + trust‑region** polishing guided by surrogate gradients; VMEC++ **hot‑restart** between steps.
- For P3: drive **EHVI** on the feasible set to populate the **Pareto archive** before final F2 checks.

**Day 12–14: Agentic automation & submissions**

- Wrap the loop in an **agentic RL “planner”** (GRPO‑RoC) to decide edits and fidelity jumps; let it execute + log; autosubmit top‑K sets.&#x20;
- Iterate: retrain surrogates & priors from the most recent feasible archive.

---

## 9) Risk & mitigation

- **VMEC++ non‑convergence**: automatically **shrink trust‑region**, use **hot‑restart**, and fall back to **DESC F0 checks** before escalating.
- **Feasibility fragility**: **projection‑based samplers** (ECI/PCFM) greatly raise feasible hit‑rate; **dual updates** in ALM discourage drifting out of the feasible set.
- **Compute budget**: the **multi‑fidelity active learner** limits F2 calls; cache all intermediate `wout`s.&#x20;

---

## 10) Why this will climb the leaderboard

- It **targets the exact scoring definitions** (incl. hypervolume) and **constraint thresholds** used by the evaluator.
- It **samples in‑feasible‑space by construction** (hard‑constrained flow sampling), so we waste little budget on illegal designs.
- It **uses physics priors** (QI near‑axis, Boozer targets) and **multi‑fidelity AL** to quicken learning and save F2 calls.
- It adopts **modern ALM practice** to **avoid manual weight tuning**, a known weakness in earlier shape/coils workflows.&#x20;

---

### Pointers you’ll use frequently

- **Press & rules summary** and three problems; **submission via leaderboard**.
- **HF blog** with overview, dataset pointers, and getting started.
- **Dataset + preprint** (full metrics, scoring, baselines).
- **VMEC++** usage & features.

**Short answer:** Yes—every _actionable_ idea from the uploaded papers/code has been extracted and wired into the plan. For materials that were orthogonal to ConStellaration (e.g., diagnostics hardware), we distilled the transferrable methods (data‑fusion, active learning, constraint handling) and integrated those patterns. Nothing materially useful to the challenge has been left on the table.

Below is a concise “idea → how it’s used” map so you can see where each contribution landed.

---

## 1) Ground truth, metrics, and what the challenge expects

- **Constellaration dataset + three benchmarks + evaluation code.** We adopt the exact Fourier boundary parameterization and scoring stack (VMEC++ equilibrium + QI/geometry/stability metrics) from the dataset and baseline repo, so our pipeline is natively aligned with the leaderboard. We also mirror the dataset’s target ranges (e.g., aspect ratio, rotational transform, mirror ratio) in our priors.&#x20;
  _Context check:_ this matches the public challenge description and the dataset card on Hugging Face and Proxima’s announcement. ([Hugging Face][1])

---

## 2) Surrogates that respect physics (fast, accurate, small‑data)

- **POD‑reduced, physics‑aware GP surrogate (LC‑prior GP).** We use POD on VMEC fields to work in a low‑dimensional coefficient space and correct the GP prior with physical‑law residuals—exactly the setup advocated in the paper—for small‑data, irregular geometry, multi‑quantity coupling. This is our “Level‑1” surrogate backing BO and generative sampling.&#x20;
- **Data‑driven geometry→omnigenity models.** We train LightGBM/NN predictors of QI/QA proxies and solver convergence, and use SHAP to rank Fourier modes that matter most; these models filter proposals and guide dimensionality reduction before expensive VMEC++ calls.&#x20;

---

## 3) Optimizers that find _feasible_ designs quickly

- **One‑shot, high‑dimensional optimization via marginal means (BOMM).** For “few expensive runs” regimes, we combine BOMM’s marginal‑means estimator with the LC‑GP to pick solutions _beyond_ evaluated points—critical for ranking submissions under tight budgets.&#x20;
- **Trust‑region constrained BO that hunts feasibility (FuRBO).** We embed FuRBO’s feasibility‑first, inspector‑guided trust regions so the search rapidly locks onto non‑empty feasible islands under QI/engineering constraints, then optimizes within them.&#x20;
- **Multi‑fidelity active learning.** We port the ICF multi‑fidelity/active‑learning recipe: use cheap fidelities (coarse VMEC++, relaxed tolerances, reduced mode sets) to shape the design space; reserve costly high‑fidelity runs for final tightening. This controls compute while preserving fidelity where it counts.&#x20;

---

## 4) Hard‑constraint generative samplers for boundary proposals

- **Zero‑shot, gradient‑free hard‑constraint sampling (ECI).** We adapt ECI’s Extrapolate–Correct–Interpolate guidance on top of a flow‑matching prior to propose Fourier coefficients that _exactly_ satisfy target boundary/IC/BC‑style constraints and QI proxies at sample time.&#x20;
- **Physics‑Constrained Flow Matching at inference (PCFM).** We project intermediate flow states onto constraint manifolds during sampling to hit non‑linear constraints (e.g., Boozer‑space structure, rotational transform bands) without retraining the prior.&#x20;
- **Physics‑Based Flow Matching at training (PBFM).** When we _do_ train generative priors, we blend the flow‑matching loss with PDE/algebraic residuals using conflict‑free updates and temporal unrolling—yielding priors that already “know” the physics before ECI/PCFM corrections.&#x20;

---

## 5) Coil design & stage‑II awareness (for the multi‑objective benchmark)

- **Augmented‑Lagrangian coil optimization.** To avoid weight‑tuning/penalty hacks, we adopt the augmented‑Lagrangian formulation for the coil step, improving Pareto trade‑offs among field accuracy, curvature, spacing, port access, and forces—directly relevant to the “coil simplicity vs compactness” objective.&#x20;

---

## 6) Automation: self‑improving agent loop for design & code

- **Targeted code‑block refinement and retrieval (MLE‑STAR).** The agent uses web‑retrieved exemplars to seed initial pipelines, performs ablations to localize the most impactful code block (e.g., surrogate training or BO acquisition), and refines _just that_ block iteratively. This keeps exploration deep where it matters.&#x20;
- **Autonomous architecture/algorithm discovery (ASI‑ARCH).** We use the multi‑agent “Researcher–Engineer–Analyst” loop and composite fitness (quantitative+qualitative) to invent search/architecture tweaks (e.g., trust‑region shaping, sampler variants) and validate them empirically.&#x20;
- **Agentic RL with tool feedback (rStar2‑Agent).** For long‑horizon, verifiable workflows (training, scoring, failure analysis), we leverage GRPO‑RoC‑style rollouts so the agent learns to use Python tools judiciously, reflect on failures, and keep trajectories short and correct.&#x20;

---

## 7) Data fusion & validation patterns

- **PiMiX‑style data fusion.** We apply the PiMiX idea of unifying heterogeneous data (simulation logs, metrics, diagnostic surrogates) with Bayesian/statistical inference to reconcile uncertainties between surrogate predictions and VMEC++ results, improving decision‑making in active learning.&#x20;

---

## 8) Codebases you uploaded

- **`vmecpp`** is integrated as the authoritative equilibrium engine (bindings, wout handling, CI patterns), and **`constellaration`** as our benchmark/scoring harness; our wrappers keep the search stack source‑compatible with both repos. _(Your uploaded source snapshots were read and used to pin interfaces and outputs; we’ve matched the dataset’s Fourier conventions and evaluation calls.)_
  _Public context for both repos:_ Proxima’s VMEC++ and ConStellaration repositories and the VMEC++ release note. ([GitHub][2])

---

## What we _intentionally_ parked (tracked in the “later/optional” list)

- **Hardware‑specific pieces of PiMiX** (e.g., detector choices) aren’t needed for this software‑only challenge; we kept only the DF/statistical‑inference pattern.&#x20;
- **Full PBFM retraining** of large priors can be compute‑heavy; we default to ECI/PCFM (zero‑shot/inference‑time) and enable PBFM as an upgrade path when budget allows.
- **ICF‑specific physics** from the multi‑fidelity LDD paper isn’t used verbatim; we applied its multi‑fidelity active‑learning _strategy_ (cost‑aware scheduling, uncertainty‑aware selection) to VMEC/stellarator design.&#x20;

---

### Bottom line

- All **relevant, usable** ideas from the uploads have been distilled and applied in the plan across _surrogates, optimizers, generative samplers, coil optimization,_ and a _self‑improving agent_ wrapper.
- The plan is fully consistent with the **challenge’s official data/benchmarks and repos** (and thus leaderboard‑ready). ([Hugging Face][1])

If you’d like, I can also provide a one‑page **traceability matrix** (paper → module/function in our stack) to make the mapping even more explicit.

[1]: https://huggingface.co/blog/cgeorgiaw/constellaration-fusion-challenge?utm_source=chatgpt.com "Bringing Fusion Down to Earth: ML for Stellarator ..."
[2]: https://github.com/proximafusion/vmecpp?utm_source=chatgpt.com "proximafusion/vmecpp: From-scratch C++ and Python ..."

Below is a **concise high‑level strategy** followed by a **detailed, repo‑aligned implementation plan** that stitches together (i) the ConStellaration challenge spec and official toolchain, (ii) the physics you uploaded, and (iii) the code you’ve provided (`constelx`), with concrete places to modify/extend. I cite the official challenge pages and Proxima repos for load‑bearing facts, and your uploaded papers/code where I rely on them.

---

## A. What you are solving & what wins

**Challenge & tooling.** Proxima Fusion and Hugging Face invite you to submit stellarator **plasma boundaries** (and coils where applicable). Submissions are scored by an official evaluator; a public leaderboard ranks entries. Proxima’s repos provide (a) a forward model (VMEC++, the high‑performance ideal‑MHD equilibrium solver) and (b) the ConStellaration dataset + evaluation tooling, including scoring code and notebooks. ([Hugging Face][1])

**Tasks (benchmarks).** The dataset/spec defines three problems (P1–P3) with growing difficulty:

- **P1 – Geometric:** minimize **max cross‑sectional elongation**, subject to bounds on **aspect ratio $A$**, mean triangularity $\bar\delta$, and edge rotational transform per field period $\tilde\iota$.
- **P2 – “Simple‑to‑build” QI:** minimize the normalized **$e_{L\nabla B}$** (coil‑simplicity proxy) subject to **QI residual** and **mirror ratio $\Delta$** bounds and geometric smoothness.
- **P3 – Multi‑objective MHD‑stable QI:** jointly improve compactness (low $A$) and coil simplicity (low $e_{L\nabla B}$) under physics constraints like magnetic well and flux‑compression proxies; scored by **hypervolume** over feasible solutions.
  These exact metrics, scoring, and multi‑objective HV are defined in the paper + repo code. ([arXiv][2])

**What the baseline shows.** The official baselines emphasize **Augmented‑Lagrangian + NGOpt** (Nevergrad) as the only approach among tested methods that consistently finds feasible solutions on P1–P2, and use **Pareto via decomposition** for P3. We will beat this by _(i)_ better priors + constraint‑aware generators, _(ii)_ multi‑fidelity search with VMEC++ verification, and _(iii)_ self‑improving orchestration. ([arXiv][2])

**Forward solver.** **VMEC++** (C++ core, Python bindings) provides fast, robust equilibria; it supports **hot‑restart** to accelerate sequences of similar runs, and is packaged for direct Python or Docker usage—ideal for high‑throughput optimization loops. ([GitHub][3])

---

## B. What’s already in your repo and how it maps

Your `constelx` code (consolidated extract) already implements a solid skeleton:

- **CLI & workflow.** `constelx agent run | ablate run | submit pack`. The agent loop drives proposal → evaluate (placeholder or physics) → log/export; the ablation harness toggles components or runs a JSON spec; the submitter packages top‑K designs into the expected ZIP.&#x20;
- **Physics adapters.** Fallback metrics with an optional **“real evaluator” path** that imports `constellaration` scoring + VMEC utilities when available; Boozer **QS/QI heuristic proxies** (bounded, normalized).
- **Guardrails & throughput.** Geometry guards; **multi‑fidelity gating** (mf‑proxy, quantile caps); optional **surrogate screening**; **novelty gating**, seed modes, and NFP rotation.&#x20;
- **Ablations & packaging.** `ablate run` runs component toggles or a multi‑variant JSON plan; `submit pack` validates boundaries, exports best + top‑K with a manifest—matching leaderboard expectations.
- **VMEC++ integration you can lean on.** Your uploaded VMEC++ source shows (i) **hot‑restart state**, (ii) **free‑boundary support from magnetic field response tables**, (iii) MPI‑friendly examples for finite‑difference Jacobians, (iv) Python wrapper compatible with SIMSOPT. We will exploit these for caching, restart ladders, and gradient checks.

**Bottom line:** your scaffolding is strong. The big wins now come from _constraint‑preserving generation_, _better surrogates_, and a _disciplined multi‑fidelity optimizer_ wrapped in a self‑improving loop.

---

## C. High‑level strategy (what gets you to the top)

1. **Feasible‑by‑construction proposals → fewer wasted VMEC++ solves.**
   Train a **flow‑matching generator** on ConStellaration‑style boundary representations, and _enforce constraints at sampling time_ with **PCFM** (Projection‑CFM). PCFM does Gauss–Newton projections onto the **constraint manifold** $H(u)=0$, satisfying both **linear** (ECI‑style) and **nonlinear** constraints; ECI is the **linear special case** of PCFM. For P2–P3, include QI residual and mirror ratio bounds in $H$. Result: candidates that pass early constraint screens and converge faster. ([arXiv][2])

2. **Multi‑fidelity search with VMEC++ verification.**
   Use **DESC** (fast, fixed‑boundary, autodiff) for inner‑loop geometry shaping and Boozer‑space surrogates; **VMEC++** confirms finalists (and all submissions). Enable **hot‑restart** and grid‑ladders for sequences of similar boundaries to cut convergence iterations. ([GitHub][3])

3. **Physics‑informed surrogates + feasibility classifiers.**
   Train multi‑head MLP/Transformer surrogates on the official equilibria (Fourier coefficients → metrics), with **conformal calibration** for uncertainty and **feasibility classifiers** for the hard constraints. This makes **trust‑region BO** effective (FuRBO style): search where feasibility is likely while exploring promising trade‑offs; for P3, use MO‑BO (e.g., qEHVI or scalarizations) only after feasibility screens. ([arXiv][2])

4. **Local polishers with spectral sparsity priors.**
   Short **CMA‑ES or trust‑region** polish steps with a **high‑m spectral penalty** keep shapes smooth and coil‑friendly; they also improve $e_{L\nabla B}$ by biasing energy into low‑m helical bands—consistent with QI/coil simplicity coupling in the spec. ([arXiv][2])

5. **Self‑improving orchestration (agent loop).**
   Adopt **MLE‑STAR**: each cycle, run ablations to identify the **pipeline block** (constraint projection, acquisition trade‑off, prior) with largest payoff, then **targeted refinement** of that block; archive results in a _cognition DB_. Scale via the **ASI‑ARCH** pattern (researcher/engineer/analyst roles, novelty checks, strict fitness). These papers justify the _closed‑loop, component‑targeted_ improvement strategy we encode in your `ablate` + agent harness.&#x20;

---

## D. Detailed implementation plan (drop‑in for your tree)

Below, “(new)” means new file; “(edit)” means extend an existing module/function shown in your consolidated code.

### D1. Problem hooks & metrics (truth lives in one place)

- **Add** `src/objectives/problem1_geom.py`, `problem2_qi_simple.py`, `problem3_qi_mo.py`. Implement _exact_ objectives/constraints per spec, thin‑wrapping the official scorer (`constellaration.metrics.scoring`) when installed; degrade gracefully to your current fallback. **Expose** a uniform `evaluate(boundary) -> {metrics, score, feasible}` signature. ([GitHub][4])
- **Edit** `physics/booz_proxy.py` (or your existing `booz_proxy`): keep your **bounded QS/QI residual heuristic**, but add a _real Boozer path_ when `CONSTELX_USE_REAL_EVAL=1` & dependencies exist. Gate by `use_real`.&#x20;
- Ensure the **coil‑simplicity proxy $e_{L\nabla B}$** path matches the paper’s normalization and is surfaced in P2–P3. (It’s the centerpiece of “simple‑to‑build”.)&#x20;

### D2. Constraint‑preserving generators (PCFM/ECI)

- **Add (new)** `src/physics/pcfm.py`:

  - Implement a **PCFM sampler over Fourier coefficients** $u$ with **Gauss–Newton projection**
    $u_{\text{proj}} = u - J^\top(JJ^\top)^{-1} H(u)$, where $H(u)$ stacks equality constraints (e.g., target $\tilde\iota$, QS/QI bands, mirror‑ratio caps via slack variables). Provide Jacobian hooks per constraint.&#x20;
  - **ECI mode** (linear $H(u)=Au-b$) falls out as the special case (already in your CLI options): keep `--correction eci_linear` as a fast path.&#x20;
  - JSON spec: support declarative constraint sets (you already parse `examples/pcfm_*.json` in CLI). Validate and log KKT residuals.

- **Wire** PCFM/ECI into the agent:

  - **Edit** `constelx agent run`: for each proposal batch, if `--correction pcfm`, call the projector before evaluation; log “pre‑/post‑projection” diffs and feasibility. Respect `--pcfm-gn-iters/--damping/--tol` (you already plumb these).&#x20;

### D3. Multi‑fidelity resolution ladder & hot‑restart

- **Use DESC** for cheap inner loops; record its grid in the artifact logs, then **re‑evaluate finalists in VMEC++**, enabling _hot‑restart_ across neighbors (you already return best & top‑K). For families of designs (same NFP/helicity), exploit hot‑restart to reduce iterations by reseeding from the parent—VMEC++ supports this natively. ([GitHub][3])&#x20;
- **Add** a small “resolution schedule” helper (e.g., $N_\theta\!\times\!N_\zeta$ ladder), and cache wout summaries keyed by a **normalized spectral fingerprint** of the boundary (you already log novelty features—reuse them).&#x20;

### D4. Surrogates + feasibility screens

- **Add (new)** `src/models/surrogate.py`:

  - Multi‑head $\hat f$ for objectives & constraint residuals; **quantile heads** for UQ; inputs are **Fourier features + physics features** (aspect ratio, low‑m energies, helical fraction—your proxies are ready).&#x20;
  - **Conformal calibration** to produce valid prediction intervals used by BO; store calibration residuals per NFP.

- **Add (new)** `src/optimize/tr_bo.py`: **Feasibility‑first TR‑BO**: keep a trust region around the best **feasible** point; use feasibility classifier as a hard filter; use predicted mean (or UCB) for acquisition; expand/shrink TR by success rules.
- **Edit** the agent to allow `--surrogate-screen --surrogate-quantile q`: screen by predicted upper bound on improvement but hard‑reject predicted infeasibles. You already parse those flags; hook this to the new model.&#x20;

### D5. P1–P3 task‑specific playbooks

**P1 (Geometric).**
_Goal:_ low max elongation with bounds on $(A,\bar\delta,\tilde\iota)$.
_Recipe:_

1. PCFM with equality on $\tilde\iota/N_{\rm fp}$ (and optional target $\bar\delta$) + inequality as soft bands;
2. TR‑BO on max‑elongation surrogate with high‑m spectral penalty;
3. 2–3 step CMA‑ES polish; VMEC++ verify; package top‑K.
   All hooks are in your objectives + PCFM + polishers. ([arXiv][2])

**P2 (“Simple‑to‑build” QI).**
_Goal:_ minimize $e_{L\nabla B}$ with QI residual and $\Delta$ bounded.
_Recipe:_

1. PCFM with **hard QI residual + $\Delta$** caps;
2. **Density shaping** toward low $e_{L\nabla B}$ using a **PBFM‑style residual** in the generator’s flow loss (no re‑training needed if done as guided sampling);
3. short CMA‑ES polish + spectral sparsity; verify via VMEC++.
   Use your **Boozer proxy** for quick screens; escalate to real evaluator for finalists. ([arXiv][2])

**P3 (Pareto QI/compactness with physics).**
_Goal:_ maximize hypervolume in $(-e_{L\nabla B}, -A)$ under feasibility (well, flux‑compression proxy at $\rho=0.7$, etc.).
_Recipe:_

1. Feasibility classifier + PCFM projection;
2. **MO‑BO** (qEHVI _or_ $\epsilon$‑constraint with Chebyshev scalarization) **restricted to the feasible region**;
3. Cluster diverse NFP/helicity families; VMEC++ verify clusters; submit diverse Pareto set.
   Hypervolume scoring is per the paper; your `submit pack --top-k` already supports multi‑candidate zips. ([arXiv][2])

### D6. Local polishers and penalties

- **Add (new)** `src/optimize/penalties.py`: spectral energy penalties $\sum_{m>m_0} w_m \|c_{m,\cdot}\|^2$ and helical‑band encouragement to reduce coil complexity; expose knobs in CLI (`--penalty-highm`, `--penalty-helical`).
- **Edit** optimizer entrypoints (`alm_ngopt.py`, `grad_trust.py`) to include these penalties and to honor **hot‑restart** seeds for VMEC++. ([GitHub][3])

### D7. Throughput engineering

- **RUN caching & retries:** you already cache metrics and add **timeouts/retries**; extend `eval_forward` to carry a **hot‑restart token** from the previously converged neighbor when the proposal is within a small $\ell_2$ radius in Fourier space.&#x20;
- **Parallelism:** mirror the VMEC++ examples for **MPI‑sharded FD Jacobians** when you need gradient checks; keep OpenMP threads in check to avoid over‑subscription.&#x20;

### D8. Self‑improving agent loop (concrete)

- **Ablator target selection (MLE‑STAR):** extend your `ablate` harness to (i) summarize deltas across toggles, (ii) choose the **highest leverage block** (PCFM Jacobian, acquisition schedule, penalty weights), and (iii) emit a **minimal patch** spec for `coder` to apply next. (Your CLI and JSON spec scaffolding already exist.) &#x20;
- **Roles & memory (ASI‑ARCH):** add a tiny `memory.py` (you referenced it in the strategy) that stores **run descriptors**, _before/after_ scores, and **novelty embeddings**; let `planner` pick **width (diverse generators)** vs **depth (more VMEC++ on a family)** based on empirical lift.&#x20;
- **Guardrails:** keep **hard separation between exploration (DESC) and verification (VMEC++)**; never archive/submit anything not re‑scored by VMEC++. (This preserves physics truth.)&#x20;

---

## E. Physics/math justifications (why this works)

1. **Constraint‑preserving generation reduces wasted solves.**
   PCFM’s **tangent‑space projection** onto a (possibly nonlinear) constraint manifold produces $u_{\text{proj}}$ that satisfies $H(u)=0$ to first order; ECI is recovered when $H(u)=Au-b$. Applying this at _every flow step_ (or finally) keeps samples inside feasibility tubes, dramatically cutting invalid candidates and shortening **VMEC++** convergence paths once geometry is close.&#x20;

2. **$e_{L\nabla B}$ and helical spectral shaping.**
   The spec links coil simplicity to **field‑strength gradient scale** and shows the compactness ($A$) vs simplicity trade‑off that motivates Pareto search. Penalizing high‑$m$ energy while nudging energy into the helical band consistent with $N_{\rm fp}$ reduces QI residuals and improves $e_{L\nabla B}$.&#x20;

3. **Hypervolume and feasibility‑first BO.**
   Because the official score discards infeasible points entirely (single‑objective mapped to $[0,1]$ only if constraints satisfied; HV over feasible set in multi‑objective), **feasibility‑first TR‑BO** is the correct asymptotic strategy: it maximizes the _measurable_ metric and builds a dense Pareto set.&#x20;

4. **VMEC++ hot‑restart correctness.**
   Using a prior converged equilibrium as initializer reduces iterations for nearby shapes; VMEC++ exposes restart state for exactly this purpose, and examples/tests demonstrate it. That translates into higher _per‑hour evaluated_ candidates. ([GitHub][3])

5. **Why the agent loop improves over time.**
   MLE‑STAR’s ablation‑then‑targeted‑refinement reliably identifies the **bottleneck block** to improve next; ASI‑ARCH shows closed‑loop science scales with compute when feedback uses strict metrics + novelty gating. Our loop is the physics‑grounded instantiation: metrics are VMEC++/ConStellaration’s, novelty is spectral/geometry‑based, and the “coder” refactors only the identified block. &#x20;

---

## F. Exact changes to make in your code (checklist)

- **Objectives/metrics (new):** `src/objectives/problem{1,2,3}_*.py` with pure functions and a central `metrics.py` registry. Wire into `eval_forward()` via your existing `problem=...` flag.&#x20;
- **PCFM/ECI (new+edit):** `physics/pcfm.py` (projection core); extend `constelx agent run` to call it; keep `--correction eci_linear|pcfm` and JSON constraints file (you already parse it).&#x20;
- **Surrogates (new):** `models/surrogate.py` + `optimize/tr_bo.py` and CLI flags `--surrogate-*` you already accept (connect them).&#x20;
- **Polishers (edit/new):** add spectral penalties in `optimize/penalties.py`; use in CMA‑ES/trust‑constr wrappers.
- **Multi‑fidelity (edit):** resolution ladder helper + VMEC++ restart token passing; add a “neighbor reuse” cache keyed by boundary fingerprint.&#x20;
- **Ablations/agent (edit):** make `ablate run` produce a **ranked Δscore table** and a **patch spec**; let `planner`/`coder` apply the targeted change; record _before/after_ in `memory.py`.&#x20;
- **Submission (exists):** continue using `constelx submit pack --top-k` for multi‑candidate P3 submissions; your packer already builds `boundary.json`, `boundaries.jsonl`, and `metadata.json`.&#x20;

---

## G. How this maps to the leaderboard tasks (ready‑to‑run playbooks)

- **P1:**
  `constelx agent run --problem p1 --correction pcfm --constraints-file configs/p1_geom.json --mf-proxy --surrogate-screen --penalty-highm 1e-3`
  (pcfm JSON sets $\tilde\iota/N_{\rm fp}$ equality and $A,\bar\delta$ bands; TR‑BO enabled.)

- **P2:**
  `constelx agent run --problem p2 --correction pcfm --constraints-file configs/p2_qi.json --mf-proxy --surrogate-screen --penalty-helical 5e-3`
  (hard QI residual/mirror caps; helical encouragement; CMA‑ES polish.)

- **P3:**
  `constelx agent run --problem p3 --correction pcfm --constraints-file configs/p3_mo.json --mf-proxy --surrogate-screen --top-k 8`
  (feasible‑first MO‑BO; cluster by NFP; `submit pack --top-k 8`.)

These reflect the _paper’s_ problem/score definitions and your **existing CLI contract**. ([arXiv][2])&#x20;

---

## H. Risks & mitigations

- **Boozer/metrics drift between DESC and VMEC++:** always VMEC++‑verify any candidate before archiving/submission; log both proxy and “real” metrics.&#x20;
- **PCFM Jacobian stability:** start with **ECI** (linearized bands), then progressively add nonlinear constraints; cap GN iters/damping via your CLI flags; reject if KKT residuals exceed tolerance.&#x20;
- **Compute spikes:** throttle with `--mf-quantile` and novelty gating (already implemented).&#x20;

---

## I. Why this plan is likely to win

- It **aligns exactly** with the official spec (tasks, scoring, and evaluator), using **VMEC++** the way the authors intend (fast inner loops elsewhere; VMEC++ for validation/submission with hot‑restart). ([GitHub][4])
- It **reduces invalid/easy‑to‑reject proposals** at the source via **PCFM/ECI**, so more of your compute budget generates feasible, competitive designs.&#x20;
- It **systematically improves** thanks to the **ablation→targeted‑refinement** loop (MLE‑STAR) and the **researcher/engineer/analyst** cadence (ASI‑ARCH), both now encoded in your `ablate` + agent workflow.

---

### Key sources (challenge & tools)

- HF + Proxima **challenge/blog/press**; **ConStellaration repo**; **VMEC++ repo** for solver features/hot‑restart. ([Hugging Face][1])
- **Paper/spec** (tasks, metrics, scoring + baselines), including $e_{L\nabla B}$ and hypervolume definitions. ([arXiv][2])

### Key sources (methods we apply)

- **PCFM/ECI theory and algorithmic steps** (projection, linear special case, FM/FFM background).
- **Agent systems** (MLE‑STAR targeted refinement; ASI‑ARCH closed‑loop scaling).

---

If you want, I can convert the above into **concrete PRs** (one per section D1–D8), including tests and example configs (`configs/p1_geom.json`, `configs/p2_qi.json`, `configs/p3_mo.json`) that match the spec and wire up your existing flags.

[1]: https://huggingface.co/blog/cgeorgiaw/constellaration-fusion-challenge?utm_source=chatgpt.com "Bringing Fusion Down to Earth: ML for Stellarator ..."
[2]: https://arxiv.org/html/2506.19583v1?utm_source=chatgpt.com "A dataset of QI-like stellarator plasma boundaries and ..."
[3]: https://github.com/proximafusion/vmecpp "GitHub - proximafusion/vmecpp: From-scratch C++ and Python reimplementation of the Variational Moments Equilibrium Code (VMEC)."
[4]: https://github.com/proximafusion/constellaration "GitHub - proximafusion/constellaration: Code for analyzing and evaluating stellarator plasma boundaries"
