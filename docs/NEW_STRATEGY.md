Below is a complete, self‑contained plan to compete for (and win) the ConStellaration Challenge, built directly from the materials you uploaded. I’ve (1) distilled what “winning” requires, (2) analyzed each paper for leverageable ideas, (3) fused the insights into a coherent technical strategy, (4) extracted concrete, reusable techniques, (5) laid out an execution plan to top the leaderboard, (6) designed a self‑improving system with LLM agents, and (7) backed the approach with math/physics/computation arguments.

⸻

0. TL;DR — Winning strategy in one paragraph

Train fast, accurate surrogates on the ConStellaration QI‑like dataset to predict all benchmark objectives/constraints, then combine a hard‑constrained generative prior (flow/rectified‑flow in function space) with a feasibility‑first trust‑region BO driver to propose plasma‑boundary shapes that are feasible immediately and steadily improve the objective. Enforce constraints exactly at inference via ECI/PCFM‑style steps; embed physics in training with PBFM to reduce residuals; guide the search with geometry features known to control turbulence (flux‑surface compression in bad curvature, geodesic curvature) so designs are not only “benchmark‑feasible” but also transport‑aware. Wrap the whole loop in an MLE‑STAR‑like agent that performs targeted ablations (parameterization, objective terms, solver settings), auto‑ensembles best ideas, and keeps improving. This joint stack attacks the three hard parts simultaneously: feasibility, sample‑efficiency, and physics quality.

⸻

1. What does it take to win? (Challenge requirements)

Benchmark substrate & rules (from the paper):
• Dataset: ~158k quasi‑isodynamic (QI)‑like stellarator plasma boundaries with vacuum VMEC++ equilibria and many figures of merit. Shapes use truncated Fourier series (stellarator symmetric), up to m,n\le 4, fixed R*{0,0}=1, giving 80 DoF in the paper’s setup. Fig. 2 shows the boundary parameterization; Fig. 6 shows metric distributions and target→outcome correlations. ￼
• Forward model: VMEC++ (C++ VMEC) to compute ideal‑MHD equilibria at R_0\simeq1 m, B_0\simeq1 T. ￼
• Three optimization problems of increasing realism (see Table 2 in the paper): 1. Geometric: minimize max elongation \epsilon*{\max} s.t. \(A\le A^\, \bar{\delta}\le \bar{\delta}^\, \tilde{\iota}\ge \tilde{\iota}^\*\). 2. Simple‑to‑Build QI: minimize e*{L\nabla B} s.t. \(\tilde{\iota}\ge \tilde{\iota}^\\), QI residual \le \(QI^\\), \(\Delta \le \Delta^\\), \(A\le A^\\), \(\epsilon*{\max} \le \epsilon*{\max}^\*\). 3. MHD‑stable QI: minimize ( -e*{L\nabla B}, A ) s.t. \(\tilde{\iota}\ge \tilde{\iota}^\\), \(QI \le QI^\\), \(\Delta\le \Delta^\\), vacuum well W*{MHD}\ge 0, and \(\langle\chi*{\nabla r}\rangle \le \langle\chi\_{\nabla r}\rangle^\\). (Symbols defined in Table 1 of the paper.) ￼

Implications for winning:
• Feasibility is hard: Multiple, often tight, constraints (topology, MHD, QI quality). Early feasible points matter (leaderboards often weight best feasible value; infeasible points don’t count).
• Expensive evaluations: VMEC++ is faster than legacy VMEC, but still costly; we must limit forward calls via data‑driven surrogates and feasibility‑first sampling. ￼
• Right inductive bias: Success depends on geometry‑aware priors and physically meaningful features.

⸻

2. What do the uploaded papers teach us? (Condensed analysis)
   • ConStellaration dataset & benchmarks: Defines the problems, metrics, dataset, and forward model. Shows outcomes strongly track targets, and reveals useful metric co‑dependences (e.g., Fig. 6 target→outcome plots). Multiple ways to generate “QI‑like” shapes are used, so the dataset captures a diverse geometry manifold. This is the bedrock for supervised surrogates and generative priors. ￼
   • Hard‑constrained generation for PDE/physics:
   • ECI sampling: gradient‑free, zero‑shot correction of pretrained flow‑matching samples via Extrapolate‑Correct‑Interpolate steps; exact satisfaction of hard constraints under projection; supports iterative mixing. Perfect to enforce equality constraints (e.g., boundary/IC‑like equalities, or linearized physics) during each sampling step. ￼
   • PCFM: Physics‑Constrained Flow Matching. General, zero‑shot framework to enforce arbitrary nonlinear constraints during flow sampling via Gauss–Newton projections and OT‑interpolant reverse updates; designed to remain aligned with the learned flow, and empirically robust for shocks/discontinuities. This extends ECI beyond simple linear projections and is ideal for QI residuals, mirror ratio caps, well constraints at the final step. Table 1 & Alg. 1–3 summarize differences and algorithm. ￼
   • PBFM: Physics‑Based Flow Matching (training‑time). Jointly minimizes flow‑matching loss + physics residual with conflict‑free gradient updates, plus temporal unrolling to improve final state accuracy and study of \sigma\_{\min} effects. Gives up to 8× lower physical residuals than vanilla FM while preserving distributional fidelity. This is the training complement to ECI/PCFM at inference. ￼
   • Feasibility‑first constrained BO at scale: FuRBO (trust‑region BO) explicitly tackles “hard‑to‑find feasible region” regimes by (i) sampling inspectors around the current best, (ii) locating likely feasible islands, (iii) adapting/shifting trust regions aggressively before sampling new points. Exactly our use‑case with expensive VMEC++ + many constraints. ￼
   • Magnetic geometry → turbulence transport: A massive NL gyrokinetic dataset shows turbulent heat flux is most controlled by (i) flux‑surface compression in bad curvature and (ii) magnitude of geodesic curvature; both correlate with heat flux in the theoretically expected direction. These become high‑value features/priors for “transport‑aware” designs. ￼
   • Data‑driven omnigenity & feature importance: A practical, low‑DoF stellarator DB + LightGBM/NN surrogates and SHAP analyses identify which boundary coefficients matter for quasi‑symmetry / quasi‑isodynamicity, and even classify solver convergence. We can reuse the feature‑importance playbook to focus search and improve solver robustness. ￼
   • Scaling laws for scientific Transformers: Training larger Transformer surrogates obeys compute/data/model scaling laws; good recipes (e.g., \mu P, Pre‑LN) improve stability. Use to choose model/data/compute trade‑offs for our surrogates/generative models. ￼
   • PiMiX (physics‑informed data fusion): Architecture for uncertainty‑aware, multi‑diagnostic fusion via neural networks and Bayesian inference; emphasizes linear inverse structure Y=MX+B. Conceptually useful for uncertainty propagation and fusing heterogeneous sources (VMEC outputs, surrogates, collisionless proxies, coil metrics). ￼
   • Self‑improving LLM agenting for ML code: MLE‑STAR: “Search + Targeted Refinement” with ablation‑driven block‑wise edits, and iterative ensembling of agent‑proposed strategies. We will adapt this to the design stack (objective terms, parameterization, solver settings, acquisition, constraint sets). ￼

⸻

3. Synthesis — The architecture we’ll build

3.1 Parameterization & features
• Primary design variables: boundary Fourier coefficients \{R*{mn},Z*{mn}\} under stellarator symmetry (as in dataset/benchmark). Start with the same mode cutoffs as the benchmark dataset to match its data manifold; optionally include phase‑shifted bases or geometrically meaningful reparameterizations (e.g., axis‑based near‑axis seeds) as alternative heads in the generator (agent will ablate). ￼
• Physics‑aware features: compute per‑surface summaries used by gyrokinetic equation (e.g., |\nabla\psi| in bad curvature, geodesic curvature magnitude, \mathbf{b}\times\nabla B\cdot\nabla x components entering v_d; Sec. 2 in Landreman et al.) and add them to the surrogate input to improve learnability and interpretability. ￼

3.2 Models
• Surrogates (regressors/classifiers):
• Multi‑task Transformer (or operator network) to predict (e*{L\nabla B},QI,\Delta,A,\epsilon*{\max},\tilde\iota,W*{MHD},\langle\chi*{\nabla r}\rangle) from boundary coefficients and derived geometry features. Use scaling‑law guidance to pick width/depth vs. dataset size. Train with uncertainty heads; calibrate with conformal predictions. ￼
• Lightweight LightGBM ensemble as a “sanity surrogate” + convergence classifier (predict VMEC++ failure/ill‑conditioning) per Laia et al.; use SHAP to rank sensitive coefficients and to warm‑start TR scaling. ￼
• Generative prior over feasible shapes:
Learn a flow‑matching (or rectified‑flow) model over the space of dataset boundaries and their physics labels.
• Train‑time physics: add PBFM residuals (e.g., soft residuals for QI, well, mirror, elongation) with conflict‑free gradient combination, plus temporal unrolling, so the final sample is already “near‑feasible.” ￼
• Inference‑time constraints: enforce hard constraints with ECI where linear projections are accurate, and PCFM steps when residuals are nonlinear/coupled (QI residual, \langle\chi\_{\nabla r}\rangle, etc.). This yields exact feasibility at the terminal sample while retaining generative diversity.

3.3 Optimizer / search driver
• Feasibility‑first TR‑BO (FuRBO):
• Initialize around high‑feasibility regions found by the constrained generator;
• Use inspector points to map the constraint isocontours and move/shrink/expand trust regions to hit feasibility fast;
• Within each TR, select candidates by Thompson sampling from the surrogate posteriors filtered by PCFM/ECI feasibility projection before VMEC++. ￼
• Physics‑in‑the‑loop selection: Penalty‑oriented multi‑objective scalarizations per benchmark (e.g., for MHD‑stable QI we minimize (-e*{L\nabla B},A) and maintain constraints); ensure W*{MHD}\ge0 and \langle\chi\_{\nabla r}\rangle cap. Transport‑aware priors: downweight proposals with high bad‑curvature compression or high geodesic curvature, since those correlate with turbulent transport; this improves downstream quality without violating benchmark targets.

⸻

4. Techniques we’ll apply (from the papers → practice)
   1. Exact constraint satisfaction at sampling time
      • Use ECI when a closed‑form/linearizable projection exists (e.g., equality‑type constraints; rotational‑transform target on a surface; linearized “mirror cap” near the boundary).
      • Use PCFM for nonlinear residuals (QI residual, well constraint, coupled inequalities): one Gauss–Newton update + OT reverse interpolant → stable, gradient‑free hard constraints.
   2. Physics‑embedded training: PBFM with conflict‑free gradient updates (g*{\text{FM}},g_R) and unrolling improves final denoised state fidelity (critical for residuals computed at t\to1). Tune \sigma*{\min} small (or zero) because added noise lower‑bounds residuals. ￼
   3. Feasibility‑driven BO: FuRBO TR updates oriented by constraint landscapes, not only objective. Excellent when feasible regions are tiny (likely here). ￼
   4. Geometry features for transport: Include |\nabla\psi| in bad curvature and |\kappa_G| (geodesic curvature magnitude) as learned penalties or tie‑breakers; they drive ITG transport and help avoid “looks‑good but transports‑bad” shapes. ￼
   5. Feature importance & convergence prediction: Use LightGBM/SHAP to rank boundary coefficients that most influence QI/QA metrics and solver failures, then restrict or re‑scale high‑impact directions in the trust region (stability), and prioritize those for agent ablations. ￼
   6. Scaling laws for surrogates/generators: Allocate compute between data/model/steps to stay on the optimal scaling frontier (e.g., for a 10× compute increase, raise dataset size ~2.5× and model size ~3.8× to get ~7× MSE reduction, per the star‑emulator scaling study). ￼
   7. Uncertainty & data fusion (PiMiX mindset): propagate uncertainties from surrogates and physics to the acquisition function and constraint margins; fuse simulator + surrogate posteriors for robust decisions. ￼

⸻

5. Concrete plan to top the leaderboard

Phase A — Baselines & infrastructure (fast)
• Reproduce paper baselines (the repo provides reference code). Validate metric computations, equilibrium scaling, and constraint checks; cache VMEC++ runs; add a convergence classifier to skip doomed regions.
• Surrogates v1: LightGBM (fast) + small Transformer multi‑task head. Validate on held‑out families (by N\_{fp}, aspect ratio bins) for OOD robustness. Calibrate uncertainties. ￼

Phase B — Constrained generative prior
• Train flow‑matching on boundary coefficients + physics labels; add PBFM residual terms on (QI,\Delta,\epsilon*{\max},W*{MHD}) with conflict‑free updates; use unrolling. After training, wrap the sampler with ECI/PCFM to hit all constraints exactly. Export a “feasible sampler” per benchmark (different constraint sets).

Phase C — Feasibility‑first optimization
• Use FuRBO with inspectors seeded from the constrained generator; inside each trust region, draw K candidates by Thompson sampling from the surrogate posteriors, project via PCFM/ECI to feasibility, then select 1–2 for VMEC++. Aggressively shift + resize TR as FuRBO prescribes to chase feasible basins. ￼

Phase D — Transport‑aware tie‑breakers
• When multiple feasible candidates tie on the official objective, prefer lower bad‑curvature compression and geodesic curvature proxies (Landreman). This doesn’t break benchmark rules but tends to improve overall physics quality and may help in the MHD‑stable QI case via correlations. ￼

Phase E — Last‑mile polishing
• Local direct search (pattern search / CMA‑ES with small steps) within feasibility to shave the objective (using the surrogate as a cheap oracle and VMEC++ sparsely).
• Ensembling candidates: Keep a Pareto set (for MHD‑stable QI); blend nearby boundary vectors and re‑project with PCFM to stay feasible while possibly improving a secondary objective (e.g., reduce A at fixed e\_{L\nabla B}). ￼

Expected benefits (why this beats a pure optimizer):
• Feasible from step 1 (constrained generator + PCFM/ECI),
• Few VMEC++ calls (FuRBO + surrogate),
• Better physics (transport‑aware features),
• Stable convergence (convergence classifier + TR).

⸻

6. A self‑improving system (agent design)

Build a “Stellarator‑STAR” agent (adapting MLE‑STAR):
• Search as a tool: scrape papers/code examples you provided (and within the benchmark repo) to assemble initial solutions: choice of parameterization (mode cutoffs, symmetry), objective scalarizations, and solver settings. (No external browsing needed to start.)
• Targeted refinement loop:
• Ablate code blocks: (i) parameterization (which (m,n) to include); (ii) physics residual weights in PBFM; (iii) ECI vs PCFM mixing steps; (iv) FuRBO TR radii/thresholds; (v) surrogate architecture.
• For each block, the agent proposes plans (e.g., “increase unrolling n and reduce \sigma\_{\min}”), implements, evaluates on the dev suite (subset of problems + unit VMEC++ calls), and keeps the winner.
• Ensemble the best blocks into an improved pipeline (the agent’s “merge” step).
• Auto‑reporting & SHAP‑driven focus: compute SHAP on LightGBM to highlight sensitive coefficients/metrics → feed the agent to focus future ablations where impact per compute is highest.

This agent structure is explicitly designed for the code‑as‑pipeline setting and mirrors MLE‑STAR’s demonstrated gains in competitive ML. ￼

⸻

7. Verification & theory (why this should work)

7.1 Hard constraints in the generator
• ECI guarantee: At each step t, extrapolate to u*1, project to the constraint manifold \hat u_1=C(u_1,G), then interpolate back along the OT path to t’\ge t. Under linear (or linearized) constraints, the projection is orthogonal and ensures exact satisfaction at t=1; iterative mixing propagates constraint information to earlier steps. ￼
• PCFM guarantee: For nonlinear h(u)=0, a Gauss–Newton step u*{\text{proj}}=u_1 - J^\top(JJ^\top)^{-1}h(u_1) moves to the tangent space of the constraint manifold (exact for affine h), then the OT displacement interpolant approximates a stable reverse update (Prop. 3.1). Repeating this each sampling step yields a final sample with zero constraint residual while staying aligned with the learned flow. ￼

7.2 Training with physics (residuals) does not collapse the distribution

PBFM uses conflict‑free gradient composition
g*{\text{update}}=(g^\top*{\text{FM}}g*v + g^\top_R g_v)\,g_v,\quad g_v=U[U(O(g*{\text{FM}},g*R))+U(O(g_R,g*{\text{FM}}))]
to ensure g*{\text{update}}\cdot g*{\text{FM}}>0 and g\_{\text{update}}\cdot g_R>0: both likelihood fit and physics residuals improve simultaneously, avoiding the collapse seen with naive weighted sums. Temporal unrolling further improves final‑time residual accuracy. Empirically reduces residuals up to 8× with competitive distributional metrics. ￼

7.3 Sample‑efficiency & feasibility
• FuRBO is provably better than global AF optimizers in narrow feasible islands: it reallocates the trust region based on constraint isocontours found by inspectors, so the expected time‑to‑first‑feasible (and thus to leaderboard‑valid entries) drops sharply. That’s key under limited VMEC++ budgets. ￼
• Surrogate learning curves (scaling laws): with compute C, dataset size N, and model size P, the star‑emulator study finds optimal co‑scaling (e.g., 10× compute → ~2.5× N, ~3.8× P → ~7× MSE drop). We reuse this to cap surrogate error such that constraint‑violation risk ≤ budgeted margin—which is essential for reliable acquisition. ￼

7.4 Transport‑aware rationale

Including bad‑curvature flux compression and geodesic curvature in the proposal/selection logic is justified because they showed the highest feature importance for ITG heat flux across >2×10⁵ nonlinear simulations and were consistent across ML models (CNNs, trees). Therefore, when options tie on the benchmark objective, picking the one with better transport proxies is physics‑sound and increases the chance the shape generalizes when constraints tighten (e.g., if organizers add transport‑relevant checks later). ￼

⸻

8. Execution details & risk controls
   • Robustness: use the VMEC++ convergence classifier to skip candidates likely to fail; inflate constraint margins based on surrogate uncertainty; re‑project by PCFM if small violations occur. ￼
   • Caching & batching: cache equilibria; batch VMEC++ calls; prioritize top‑K candidates by EI × feasibility probability inside the TR.
   • Ablations the agent should try early: (i) mode‑set restrictions (remove the most SHAP‑unstable coefficients), (ii) PBFM unrolling steps, (iii) ECI vs PCFM ratios per constraint, (iv) TR radius schedule.
   • Validation: maintain a hidden validation set of boundary seeds, report distribution of constraint slacks, and check consistency across N*{fp} subsets (dataset spans N*{fp}=1–5; Fig. 4 shows examples). ￼

⸻

9. What we’ll deliver (per benchmark)
   • Geometric: Feasible designs with lower \epsilon*{\max} than baselines, via constrained generator + FuRBO; tie‑break by lower geodesic curvature.
   • Simple‑to‑Build QI: Minimal e*{L\nabla B} while meeting (\tilde\iota,QI,\Delta,A,\epsilon*{\max}) by PCFM‑projected proposals; SHAP‑guided restriction on unstable modes.
   • MHD‑stable QI: Pareto set improving -e*{L\nabla B} and A with hard constraints W*{MHD}\ge0 and \langle\chi*{\nabla r}\rangle cap; final “blend + PCFM re‑project” to polish the frontier. ￼

⸻

Final note

The plan above is tightly tied to the dataset/benchmarks, uses physically grounded constraints (hard at inference, soft at training), exploits state‑of‑the‑art constrained generation and BO, and is wrapped in an agent that continually gets better. It is deliberately redundant (surrogate + LightGBM, ECI + PCFM) to tolerate model mismatch, and transport‑aware to avoid “paper‑wins, physics‑losses.”

Below is a complete code scaffolding and an end‑to‑end implementation plan tailored to the ConStellaration benchmarks (Geometric, Simple‑to‑Build QI, and MHD‑Stable QI). It fuses (i) the dataset/evaluator released with the challenge, (ii) constraint‑aware generative modeling (Flow Matching with hard constraints at inference and physics‑aware training), (iii) trust‑region Bayesian optimization to find feasible regions fast, and (iv) a self‑improving coding agent that does targeted code‑block refinement. Where I cite specific formulas/definitions/benchmarks, I point to the uploaded papers.

⸻

0. Why this design will win
   • Benchmarks & metrics are stable and well‑specified. The challenge releases three constrained problems, an evaluator, and strong baselines; boundaries are Fourier‑parametrized and scored through VMEC++ equilibria and derived metrics (e.g., \tilde\iota,\ A,\ \epsilon*{\max},\ \Delta,\ \mathrm{QI},\ e*{L\nabla B},\ W*{\mathrm{MHD}}, \langle \chi*{\nabla r}\rangle). We can target each metric explicitly and respect the submission interface.
   • Hard constraints during candidate generation (PCFM) + physics‑aware training objective (PBFM) give exact satisfaction at the final sample while staying faithful to the learned generative flow; we use this to sample feasible surfaces before any costly oracle call.
   • Feasibility‑first trust‑region BO (FuRBO) rapidly relocates the search to viable islands in high‑dimensional, irregular feasible sets (a known pain‑point), complementing the baselines. ￼
   • Strong surrogates that scale: we train multi‑task surrogates with Transformer‑style emulators using scaling‑law guidance to predict metrics (and feasibility) from boundary coefficients, letting us prune aggressively before VMEC++ calls. ￼
   • Self‑improving pipeline leverages an MLE agent that runs ablations to identify the code block with the largest marginal gain and refines it iteratively (model choice, latent generator, regularizers, acquisition, evaluator), not the entire stack at once. ￼
   • Physics‑informed priors: QI residual/QS proxies, vacuum‑well proxy for ideal‑MHD stability, and simple coil proxies are built in; turbulence‑safety is handled through a geometric proxy (\langle \chi\_{\nabla r}\rangle) aligned with ML findings about features that correlate with ITG heat flux.

⸻

1. Repository scaffolding (modules, APIs, run order)

Monorepo: constella/ (Python; Poetry/uv; CUDA optional)

constella/
configs/ # Hydra/YAML configs for each benchmark & run mode
data/
constellaration/ # dataset cache; splits; stats
constella*core/
**init**.py
types.py # Typed dataclasses for Boundary, Metrics, Constraints, Result
io.py # I/O for boundaries (R_mn,Z_mn), VMEC++ formats
parameterizations/
fourier.py # FourierBoundary: Θ ↦ Σ*Θ(θ,φ); enforce symmetry & Nfp
nearaxis.py # optional: near-axis seeds for QI-like fields
physics/
vmecpp.py # high-level VMEC++ driver (lo/hi fidelity)
metrics.py # computes A, εmax, Δ, QI, e*{L∇B}, W_MHD, <χ∇r>, ...
evaluator.py # wraps official scoring for all three problems
constraints/
geometric.py # C_geometric(Θ) ≤ 0
simple_qi.py # C_simpleQI(Θ) ≤ 0
mhd_qi.py # C_mhdQI(Θ) ≤ 0
projections.py # Newton/Gauss-Newton projections for constraint satisfaction
surrogates/
datasets.py # builds (Θ → metrics,feasible) training sets from ConStellaration
models/
xgb.py # LightGBM/LightGBM-LSS baselines
ffn.py # MLP baselines
transformer.py # TransformerPayne-style emulator for metrics
flow/
fm.py # functional flow matching backbone
pbfm.py # Physics-Based Flow Matching training wrapper
pcfm.py # Physics-Constrained Flow Matching sampler (inference)
training.py # unified trainer; scaling configs; CV
selection.py # model selection w/ uncertainty; conformal cutoff
optim/
acquisition.py # TS, CEI, Knowledge-Gradient variants for constrained BO
furbo.py # Feasibility-driven trust-region BO
alm_ngopt.py # Augmented Lagrangian (Nevergrad) baseline
multiobj.py # ε-constraint/Pareto support (Problem 3)
pipeline.py # Orchestrates optimization stages
generation/
latent.py # PCA/Autoencoder latent; GMM over feasible; normalizing flow
seeds.py # seed generation: near-axis, heuristic QI, dataset mined
agents/
mleastar.py # MLE-STAR agent: ablation-driven code-block refinement
dgm.py # “Darwin Gödel Machine” self-modifier (safe sandbox)
orchestration/
run_geometric.py
run_simple_qi.py
run_mhd_qi.py
submit.py
logging.py # MLFlow/W&B + JSONL
parallel.py # Ray/Dask for VMEC++ batches
scripts/
prepare_data.py
train_surrogates.py
warmstart_latent.py
optimize*{geometric|simpleqi|mhdqi}.py
sample_pcfm.py
tests/
README.md

Key API contracts (type hints)

@dataclass
class Boundary:
nfp: int
Rmn: Dict[Tuple[int,int], float] # (m,n) -> coefficient
Zmn: Dict[Tuple[int,int], float] # invariants (e.g., R00=1, stellarator symmetry) enforced in parameterizations/fourier.py

@dataclass
class Metrics:
A: float
iota_tilde: float
eps_max: float
Delta: float
QI: float
e_LnablaB: float
W_MHD: float
chi_grad_r: float # <χ∇r>
feasible: Dict[str,bool] # per-problem constraints

class ForwardModel(Protocol):
def solve(self, b: Boundary, fidelity: str) -> Equilibrium: ...

def evaluate(boundary: Boundary, problem: str, fidelity: str) -> Tuple[Metrics, float]:
"""Return metrics and objective value; objective defined by problem spec."""

Run order (per problem) 1. prepare*data.py → cache dataset & evaluator schema. ￼ 2. train_surrogates.py → train meta‑models (metrics + feasibility classifiers). ￼ 3. warmstart_latent.py → pretrain feasible generators (PCA/GMM + normalizing flow; supervised AE for QS/QI manifolds if desired). ￼ 4. optimize*{geometric|simpleqi|mhdqi}.py → 3‑stage optimizer (PCFM/PBFM sampling → FuRBO BO → ALM polishing) with multi‑fidelity oracle control. 5. submit.py → pack final set of boundaries in the evaluator’s accepted representation (truncated Fourier series). ￼

⸻

2. End‑to‑end implementation plan (step‑by‑step)

Stage A — Ground truth & evaluator fidelity

A1. Mirror the official scoring. Wrap VMEC++ in physics/vmecpp.py with two presets:
• Hi‑fidelity = evaluator settings; used for final scoring/selection.
• Lo‑/Mid‑fidelity = reduced flux‑surface count & tolerances for screening (10–20× cheaper). The paper explicitly used lower‑fidelity VMEC++ during dataset generation and higher fidelity for scoring; we match that pattern. ￼

A2. Implement all metrics exactly as defined (and unit‑normalize as in spec):
• Geometric problem objective: minimize \epsilon*{\max} with constraints on A, \bar\delta, \tilde \iota. ￼
• Simple‑to‑build QI objective: minimize e*{L\nabla B} (normalized magnetic gradient scale length) with constraints on \tilde\iota, QI residual, mirror ratio \Delta, aspect ratio A, max elongation \epsilon*{\max}. Clarify: QI residual integrates deviation from a precise QI field, and e*{L\nabla B} is the coil‑simplicity proxy.
• MHD‑stable QI: bi‑objective (-\ e*{L\nabla B},\ A) with constraints: \tilde\iota, QI, \Delta, vacuum magnetic well W*{\mathrm{MHD}}\ge 0, and turbulence proxy \langle\chi\_{\nabla r}\rangle bound.

A3. Unit tests against released baseline numbers and Pareto front examples to ensure parity. (The paper shows baselines and a sparse Pareto front; we should see comparable values at equal settings.)

⸻

Stage B — Surrogates (prune VMEC++ calls)

B1. Datasets. Build supervised sets \mathcal{D}=\{(\Theta, y)\} from ConStellaration (Fourier Θ → metrics, feasibility per problem). Include the paper’s feasible‑domain mining approach (PCA + RF + GMM) as a baseline generator.

B2. Fast baselines. LightGBM/LightGBM‑LSS for regression/classification of metrics & VMEC convergence (the Laia/Jorge/Abreu paper shows tree models & SHAP were effective). Use these to screen out blatant non‑feasible points. ￼

B3. Scalable emulator. Train a multi‑task Transformer emulator (\Theta\!\mapsto\! all metrics + feasibility), with µP scaling rules to trade off data/model/compute optimally; this improves domain transfer and avoids plateaus. Target the loss portfolio to match metric scales and add calibration heads for uncertainty. ￼

Why this matters: The challenge’s baselines show that gradient‑free ALM found feasible solutions but burned large function calls; we reduce calls by only forwarding high‑probability feasible candidates. ￼

⸻

Stage C — Constraint‑aware generative modeling

C1. Training (PBFM). Train a flow‑matching model in functional space (over boundary coefficients Θ or over Σ(θ,φ)) with a physics residual loss added without conflict using ConFIG update (so gradient directions for generative + physics losses align). Compute residual terms corresponding to the benchmark’s constraints (e.g., QI residual, coil simplicity prior, inequality slack approximations). Unroll to refine the terminal prediction used in residuals. ￼

C2. Sampling (PCFM). At inference, interleave forward flow steps with Gauss–Newton projections that hard‑enforce equality/inequality constraints (use log‑barriers into equalities when needed, or projection on active set). Use the OT displacement interpolant for stable reverse updates (per PCFM). Result: terminal samples satisfy constraints exactly, ready for lo‑fidelity VMEC++ scoring. ￼

C3. Latent preconditioning. Warm‑start the flow from a feasible region via PCA/AE latents (Laia et al. report supervised AEs that align latent axes with QS/QI). This improves hit‑rate for feasible QI/QS. ￼

⸻

Stage D — Global search: Feasibility‑first BO + ALM polishing

D1. FuRBO for feasibility islands. Start around best‑predicted feasible Θ (from surrogates or PCFM). Sample inspectors in a ball, rank by constraint satisfaction probability ⁺ objective from surrogates, then shift/resize the trust region to where feasibility seems densest. Within TR, do Thompson sampling on objective under feasibility. (FuRBO is designed for the “hard‑to‑find feasible” regime.) ￼

D2. Oracle ladder. Each FuRBO batch: (i) surrogate pre‑filter → (ii) PCFM projection (hard constraints) → (iii) lo‑fidelity VMEC++ → (iv) only top‑k pass to hi‑fidelity scoring.

D3. ALM refinement. Take the top 10–20 candidates and run a short non‑Euclidean proximal ALM block (Nevergrad NGOpt sub‑solver, as in the baseline) with a trust‑region proximal term (shown essential by the paper), to shave last constraint violations or improve objective. ￼

D4. Problem‑specific tactics
• Geometric: exploit cheap geometry constraints; DESC/JAX auto‑diff can help for rapid local steps (their generation used DESC + ALM; we only use auto‑diff locally to shape geometry measures). ￼
• Simple‑to‑Build QI: bias sampling toward low‑QI residual manifolds (PCFM training condition uses QI residual), while the objective e*{L\nabla B} trades coil simplicity; keep A,\ \epsilon*{\max},\ \Delta as active inequality constraints in PCFM. ￼
• MHD‑Stable QI: ε‑constraint the aspect ratio to sweep the Pareto front as in the paper; integrate the W*{\mathrm{MHD}} and \langle\chi*{\nabla r}\rangle constraints into the PBFM residual and PCFM projector. (The turbulence proxy is aligned with ML evidence that flux‑surface compression in bad curvature correlates with higher ITG heat flux; we cap it.)

⸻

Stage E — Self‑improving LLM agent loop (targeted refinement)

E1. Ablation‑guided targeting (MLE‑STAR). Every N optimizer iterations, run an ablation study over pipeline modules (surrogate type, PBFM weights/unrolling, PCFM step size, FuRBO TR radius/inspectors, ALM penalty schedule, seed generator). Select the single block with highest marginal impact to refine next; propose K concrete plans and implement/evaluate them in parallel. Keep the best. ￼

E2. Ensemble of solutions. Maintain diverse candidates (different Nfp, seeds, latent manifolds). For submissions needing a single design, pick the best objective‑under‑constraints. For multi‑objective (Problem 3), publish a sparse Pareto set.

E3. DGM safety‑box. Allow the agent to propose code rewrites in a sandbox; only merge after unit tests (metrics parity, evaluator parity) pass.

⸻

3. What each benchmark needs (tactics)

Problem 1 — Geometric (min \epsilon*{\max} s.t. A, \bar\delta, \tilde{\iota})
• Seed with near‑axis or rotating ellipses; project to satisfy A,\bar\delta,\tilde\iota exactly via PCFM, then minimize \epsilon*{\max}. ￼
• Use FuRBO with a tight TR (this is a smooth geometry surface) and small inspector sets; mostly single‑objective BO + occasional local ALM.
• Expect fast wins relative to baselines (SciPy trust‑constr struggled to produce feasible results). ￼

Problem 2 — Simple‑to‑Build QI (min e*{L\nabla B} s.t. \tilde\iota,\ \mathrm{QI},\ \Delta,\ A,\ \epsilon*{\max})
• QI residual \mathrm{QI} and e*{L\nabla B} are antagonistic in parts of the space (QI wants elongated/large A; coil simplicity wants the opposite). Bake both into PBFM during training; at sampling, enforce all constraints via PCFM and treat the objective via BO. Definitions: e*{L\nabla B} is the normalized magnetic gradient scale length proxy for coil simplicity; \mathrm{QI} is an integral residual vs a QI target field. ￼
• Use feasibility‑first FuRBO to locate low‑QI feasible islands, then optimize e\_{L\nabla B}. This should beat the ALM‑only baseline that required substantial compute. ￼

Problem 3 — MHD‑Stable QI (minimize (-e*{L\nabla B}, A) s.t. \tilde\iota,\ \mathrm{QI},\ \Delta,\ W*{\mathrm{MHD}}\ge 0, \langle\chi*{\nabla r}\rangle \le cap)
• Treat \langle\chi*{\nabla r}\rangle cap as a strict constraint in PCFM (hard project each step), and use ε‑constraint on A to trace a Pareto set as in the paper’s example. The turbulence proxy choice aligns with ML findings (flux‑surface compression in bad curvature & geodesic curvature magnitude correlate with heat flux); keeping it bounded avoids pathological turbulence.

⸻

4. Concrete algorithms & settings

PCFM sampler (inference):
At each flow step \tau\to \tau+\Delta\tau: quick forward solve to \tau=1; compute residuals h(u*1) for active constraints; one Gauss–Newton projection u*{\text{proj}}=u_1 - J^\top(JJ^\top)^{-1}h; reverse via OT displacement to \tau’ to continue. Use inequality‑to‑equality via slack variables or log‑barriers on active set. (Final state exactly satisfies h(u_1)=0.) ￼

PBFM trainer:
Total loss L = L*{\text{FM}} + L*{\text{phys}} with conflict‑free update g*{\text{update}} (ConFIG); temporal unrolling to refine the terminal prediction used in computing residuals; consider stochastic sampler during training for better distributional fit. Use \sigma*{\min}=0 or very small due to physics residual sensitivity. ￼

FuRBO loop:
Maintain TR \mathcal{B}(x^\*,R); draw inspectors uniformly, rank by feasibility‑prob × surrogate objective; place/shape TR from top‑P% inspectors; sample candidates in TR by TS on objective under feasibility; evaluate via oracle ladder; adapt R by success/failure counters. ￼

ALM polishing:
Use the paper’s non‑Euclidean proximal ALM (Nevergrad NGOpt as sub‑solver) with anisotropic scaling of Θ; a short budget (e.g., 5e3 evaluations lo‑fidelity + 100 hi‑fidelity) per top candidate. ￼

Surrogate scaling guide:
Given a 10× training‑compute increase, aim for ~2.5× data and ~3.8× model size to remain on‑law and achieve ~7× MSE drop (guidance from scaling law measurements). Use µP parameterization, Pre‑LN residual blocks, and RMSNorm. ￼

⸻

5. Submission pipeline (CLI)
   • Geometric:
   python scripts/optimize_geometric.py +configs/geometric.yaml --stages PCFM,FuRBO,ALM --parallel 64
   • Simple‑to‑Build QI:
   python scripts/optimize_simpleqi.py +configs/simpleqi.yaml --stages PCFM,FuRBO,ALM
   • MHD‑Stable QI:
   python scripts/optimize_mhdqi.py +configs/mhdqi.yaml --pareto_sweep A_targets=[6,8,10,12]
   • Pack & submit:
   python orchestration/submit.py --select 'best_feasible' --problem simpleqi

Config contains: N\_{\text{inspectors}}, TR radius, PBFM unrolling steps, PCFM \Delta\tau, projection tolerance, surrogate ensemble choices, oracle ladder quotas (lo/hi fidelity).

⸻

6. Self‑improvement plan (agent loop)
   • Every 2–4 hours: run agents/mleastar.py ablation on: (i) flow sampler projection tolerance; (ii) TR geometry; (iii) surrogate ensemble; (iv) latent manifold; (v) ALM penalties; (vi) seed types. The agent proposes K targeted refinements for the single most impactful block, implements them, evaluates, and merges best. ￼
   • Archive trajectories with full metadata; keep a diversity pool for multi‑objective Problem 3.
   • Safety: all agent edits gated by tests that check evaluator parity (metrics match definitions) and constraints (no leakage/tampering).

⸻

7. Theoretical verification & risk controls
   • Hard constraints at sample end: PCFM’s projection step guarantees exact satisfaction at the final time (affine constraints provably exact; nonlinear handled via Gauss–Newton to tangent manifold, iterated each step). This is precisely what we need before oracle calls. ￼
   • No gradient tug‑of‑war during training: PBFM uses conflict‑free update so physics residuals and flow‑matching both improve per step; temporal unrolling stabilizes final prediction used in residuals. ￼
   • Feasibility discovery complexity: FuRBO centers TR using constraint surrogates and inspectors; this adaptively relocates search to feasible islands, avoiding dead budgets common in constrained BO at high‑D. ￼
   • Surrogate reliability: transformer scaling laws + uncertainty ensembling and conformal thresholds; miscalibrated surrogates are buffered by PCFM hard constraints and the oracle ladder. ￼
   • Physics alignment: turbulence proxy \langle\chi\_{\nabla r}\rangle is consistent with interpretable ML findings (flux‑surface compression in bad curvature & geodesic curvature magnitude ↑ → higher ITG heat flux). Bounding it promotes turbulence‑robust designs. ￼
   • Baseline parity: We keep ALM‑NGOpt as a polishing stage because it was the only baseline consistently finding feasibility across the challenge’s examples; our stack strictly dominates by adding PCFM/PBFM/FuRBO and better surrogates. ￼

⸻

8. Day‑by‑day execution (for the GPT‑5 coding agent)

Day 1–2:
• Implement vmecpp.py, metrics.py, evaluator.py with unit tests that reproduce the paper’s metrics (and any example values in the repo). ￼
• Build LightGBM surrogates; quick AE/PCA+GMM feasible sampler; baseline ALM‑NGOpt.

Day 3–4:
• Implement PBFM training loop with physics residuals matching each benchmark; implement PCFM sampler. Smoke tests: constraints exactly zero at sampler end on synthetic constraints; then on small real batches.
• Implement FuRBO TR updates & inspectors; plug oracle ladder.

Day 5–6:
• Integrate the three‑stage optimizer; do ablations (agent) to lock in projections and TR policy; run full pipeline for Problem 1 and 2; compare to baseline tables. ￼

Day 7+:
• Multi‑objective sweeps for Problem 3; publish a sparse Pareto set similar to paper (aspect‑ratio ε‑constraints). ￼

⸻

9. Notes aligning with the papers (so nothing is missing)
   • Problem definitions & evaluator: three problems with explicit objectives/constraints, Fourier representation, VMEC++ oracle hierarchy, and baseline ALM‑NGOpt.
   • QI/QS proxies and coil simplicity proxy e\_{L\nabla B} (Kappel et al.) are implemented per definitions; mirror ratio/elongation/aspect ratio as constraints; MHD vacuum well and turbulence proxy per spec. ￼
   • Feasible generative models: ConStellaration already demonstrates generative modeling of feasible domains (PCA+RF+GMM); we replace RF with flows + PCFM/PBFM for stronger, constraint‑consistent sampling. ￼
   • QS/QI latent structure: supervised AE shows clean structure—useful for preconditioning flows & BO starts. ￼
   • ITG turbulence geometry levers used in Problem 3’s proxy are consistent with interpretable ML findings (flux‑surface compression in bad curvature and geodesic curvature magnitude). ￼

⸻

Minimal example of a run (Simple‑to‑Build QI) 1. Sample 1k Θ via PCFM (all constraints as equalities with small slacks), evaluate lo‑fidelity → keep top‑100 feasible lowest \mathrm{QI}. ￼ 2. Start FuRBO at best Θ; inspectors=512; TR radius auto; do 30 batches, each 64 candidates; hi‑fidelity for top‑3 per batch. ￼ 3. ALM polish top‑10 for 5e3 evals lo‑fidelity + 100 hi‑fidelity; select best feasible minimal e\_{L\nabla B}; submit. ￼

⸻

If you’d like, I can adapt this into a ready‑to‑code task list (issues/PRs) for the GPT‑5 coding agent, including function signatures and concrete defaults for each config.
