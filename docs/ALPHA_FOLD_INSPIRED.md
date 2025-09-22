Thanks for the six new papers. Quick heads‑up: some earlier uploads have expired on my side. I’ve pulled the six new papers from their public sources and analyzed them below. If you’d like me to re‑read any of the expired earlier uploads from local files, please re‑upload them.

⸻

What you asked me to do 1. Analyze all six papers. 2. Extract ideas that improve our ConStellaration plan and the self‑improving system. 3. Piece everything together. 4. Produce a leaderboard‑winning plan. 5. Produce a plan for recursive self‑improvement. 6. Verify with theory (math/physics/computation) and present the refined plan.

⸻

A. What each paper contributes (actionable takeaways)

A1) Darwin Gödel Machine (DGM): open‑ended self‑improving code agents

Core idea. Keep an archive of agents; repeatedly mutate an archived agent (using a foundation model) to generate variants; validate each variant on objective benchmarks; keep the best; repeat. The system improves its own ability to self‑improve (bootstrapping edits like better editing tools, patch validation, history of attempts). Result: large gains on code benchmarks through open‑ended exploration with strict safety (sandboxing & gating). ￼
Use for us. Wrap our entire stellarator‑design stack in a DGM‑style self‑modifier: archive pipelines (configs + code modules), generate code/param changes, auto‑test on a dev leaderboard (proxy problems), and keep winners. (See Section D.)

⸻

A2) DeepSeek‑R1 (Nature): RL that induces reasoning

Core idea. A pure‑RL pipeline (GRPO variant of PPO) with rule‑based, verifiable rewards on math/coding; later blended with model‑based preference rewards. It shows thinking‑time scaling emerges naturally, and performance leaps when the reward is verifiable (compiler tests, exact numerical answers). ￼
Use for us. Stellarator design has an exact verifier (the evaluator + VMEC++ constraints). We can run reasoning‑RL over the agent’s design decisions (e.g., constraint sets, acquisition hyper‑params, projection tolerances, unrolling depth) with rule‑based rewards (feasible? improved objective? wall‑clock budgets?). This makes our LLM agent learn better “research moves” over time—no human labels needed, because feasibility and metric improvements are fully checkable. (See Section D/E.)

⸻

A3) Toward a Physics Foundation Model (PFM)

Core idea. A single General Physics Transformer trained on a large, heterogeneous simulation corpus can learn dynamics from context and generalize across domains (fluids, shocks, multi‑phase, thermal) zero‑shot by in‑context inference—no explicit PDE is given to the model. ￼
Use for us. Build a multi‑physics surrogate that ingests boundary coefficients + a minimal set of geometry/field diagnostics and predicts our metrics (QI residual, mirror ratio, well, etc.). Use in‑context prompts to swap constraint regimes (Geometric / Simple‑QI / MHD‑QI) without retraining, and to adapt to novel aspect‑ratio or N\_\mathrm{fp} sub‑domains. (See Section C1/C2.)

⸻

A4) Discovery of Unstable Singularities

Core idea. ML‑assisted search, high‑precision Gauss–Newton, and special training schemes to discover unstable, self‑similar blow‑up solutions, with numerical precision down to double‑float round‑off. They report an empirical law linking blow‑up rate to an “order of instability.” The workflow combines curated architectures and Gauss–Newton projections to extremely tight residuals. ￼
Use for us. Upgrade our PCFM projection to a high‑precision Gauss–Newton with trust‑region, Tikhonov damping and condition‑number aware step control. Track the smallest singular value of the active‑constraint Jacobian as an “order‑of‑instability” analogue to decide (i) step size; (ii) when to switch to BO; (iii) which coefficients to freeze. This will stabilize feasibility projections near ill‑conditioned corners of the feasible set. (See Section C3.)

⸻

A5) AlphaFold (Nature 2021): end‑to‑end, physics‑aware, equivariant network with “recycling”

Core idea. Inject geometric inductive bias (Invariant Point Attention), pair/single representations, and recycling (iterative refinement) in an end‑to‑end model trained with self‑distillation; yields near‑atomic accuracy with calibrated uncertainties. ￼
Use for us.
• Build equivariant surrogate heads with SO(2) \times \mathbb{Z}{N\mathrm{fp}} invariance for surfaces; add recycling passes to iteratively refine predicted fields (e.g., QI residual on surfaces) and calibrated uncertainty akin to pLDDT for constraint slack.
• Use pairwise “surface‑point” features (pairs of (\theta,\phi)) similar to pair reps to better learn global geometric couplings. (See Section C1.)

⸻

A6) Deep Loop Shaping (Science 2025): frequency‑domain RL for physical control

Core idea. Frequency‑domain reward shaping and RL yielded 30–100× reduction of control noise at LIGO (10–30 Hz), realized on the real Livingston interferometer. It proves rewarding the right spectral features can tame hard physical loops. ￼
Use for us. Define spectral rewards over Fourier content of constraint residuals across (m,n) modes (e.g., QI residual spectrum, mirror‑ratio ripple, field‑line curvature ripple). Then run loop‑shaping RL to suppress “bad bands” in the geometry response—this is an extra handle beyond scalar constraints. (See Section B2/D2.)

⸻

B. How these improve our current ConStellaration plan

We keep our core stack (physics‑embedded flow generation + PCFM hard constraints + FuRBO feasibility‑first BO + ALM polish), and add four upgrades:

B1 — Self‑improving research agent 2.0 (DGM + R1).
• Replace the earlier “MLE‑STAR‑style” agent with a DGM archive of pipelines (code + configs). Each iteration: sample pipeline → propose K edits (via LLM) → compile & sandbox → evaluate on rule‑verified dev leaderboard (subset of the three problems) → archive Pareto‑improving variants. Rewards are fully rule‑based (feasible? better objective? fewer VMEC++ calls?), in the DeepSeek‑R1 spirit. ￼

B2 — Frequency‑shaped objectives (Deep Loop Shaping).
• Add a spectral penalty to our training‑time residuals (PBFM) and inference‑time projections (PCFM):
J*{\text{spec}}(\Theta)=\sum*{(m,n)}w\_{mn}\,\| \widehat{r}{mn}(\Theta)\|2^2,
where r is, e.g., the QI residual over the surface and \widehat{\cdot} denotes the surface Fourier transform. Choose w{mn} where ripple is known to harm transport or feasibility. Also apply RL loop‑shaping on w{mn} itself to concentrate effort where it buys the most. ￼

B3 — Better surrogates: PFM‑style multi‑task emulator + AlphaFold‑style heads.
• Introduce a GPhyT‑like backbone that can in‑context switch among the three problem regimes and across N\_\mathrm{fp}, with equivariant heads and recycling passes to refine predicted fields & constraints (like pLDDT, we produce calibrated slack estimates). ￼

B4 — High‑precision Gauss–Newton PCFM.
• Upgrade PCFM’s projection to trust‑region Gauss–Newton with Tikhonov damping; monitor \sigma\_{\min}(J) of the active‑constraint Jacobian to set step size and decide when to freeze sensitive modes. This mirrors the unstable‑singularity playbook for handling ill‑conditioned residuals at round‑off limits. ￼

⸻

C. Piece everything together (system design)

C1) Surrogates (PFM + AlphaFold inductive bias)
• Backbone: a transformer taking boundary coefficients \Theta plus a surface sampling of geometric primitives (curvature, geodesic curvature, compression in bad curvature) and optional coil proxies.
• Equivariant heads: impose SO(2) (poloidal rotation) and \mathbb{Z}{N\mathrm{fp}} (toroidal periodicity) equivariance; predict per‑surface fields then integrate to metrics.
• Recycling: iterate 2–4 times, each pass refining fields, like AlphaFold’s structure module recycling.
• Calibration: output mean + log‑variance for each metric + constraint‑feasibility logits; conformalize on held‑out families. ￼

C2) Training (PBFM + spectral losses)
• Loss: L=L*{\text{FM}}+\lambda_1 L*{\text{phys}}+\lambda*2 J*{\text{spec}}. Conflict‑free gradient composition (PBFM) avoids tug‑of‑war between generative fit and physics residuals; temporal unrolling improves terminal‑state accuracy. ￼
• Spectral residuals: penalties on \widehat{r}\_{mn} harmonics for QI residual, mirror ripple \Delta, elongation ripple, etc., following Deep Loop Shaping’s “reward where it matters in frequency.” ￼

C3) Generation (PCFM with robust Gauss–Newton)
• Active set: translate inequalities to equalities with slacks; build Jacobian J of active constraints.
• Step: \delta = \arg\min*\delta \|J\delta + r\|\_2^2 + \alpha\|\delta\|\_2^2 (Tikhonov); trust radius clips \|\delta\|.
• Stability index: monitor \sigma*{\min}(J) (like “order‑of‑instability”); if small, decrease trust radius, increase damping \alpha, and freeze top‑SHAP modes for a few iterations. ￼

C4) Search (FuRBO + spectral shaping + ALM)
• Feasibility‑first TR‑BO (FuRBO): center trust region where feasibility probability is high; inspectors map constraint isocontours; sample within TR via Thompson sampling on objective conditioned on feasibility.
• Spectral‑aware tie‑breakers: candidates with lower bad‑curvature compression and geodesic curvature (transport proxies) win ties—aligns with turbulence findings we already use.
• ALM polish: short, anisotropic ALM (Nevergrad/NGOpt) on the top K designs. (This is our established last mile.)

⸻

D. A plan to win the leaderboard

D1 — Fast feasible starts.
Use PCFM with robust Gauss–Newton to generate already feasible candidates for each problem set (Geometric / Simple‑QI / MHD‑QI) before any VMEC++ call. That wins the time‑to‑first‑feasible race and quickly sets a high bar.

D2 — Shrink the VMEC++ bill.
The PFM‑style surrogate with recycling + calibration filters candidates; FuRBO only promotes likely‑feasible shapes; the oracle ladder (lo‑fidelity → hi‑fidelity) keeps budgets tight.

D3 — Spectral shaping for extra gains.
Add frequency‑domain penalties to suppress harmful geometry ripple harmonics (mirror & QI residual bands). Deep Loop Shaping shows that rewarding frequency content can produce outsized physical improvements; we reuse this principle on the geometry spectrum. ￼

D4 — Local polish & blending.
Run short ALM and blend‑then‑re‑project (PCFM) the top designs to nudge the objective without breaking feasibility; publish a tight Pareto set for MHD‑QI.

D5 — Continuous improvement loop.
Every few hours, the DGM agent proposes code/config mutations (e.g., new spectral weights, TR radii, which modes to freeze, PBFM unrolling steps). R1‑style rule‑rewards (feasible? better objective? fewer calls?) pick winners. This ensures leaderboard creep in our favor. ￼

⸻

E. A plan for a recursively self‑improving system

E1 — Archive & lineage.
Maintain an archive of whole pipelines (git+config hashes). Each entry stores: objective numbers per problem, budget spent, constraint slacks distribution, spectral fingerprints, convergence rates.

E2 — Proposal generators (LLM “research moves”).
Three move types, all rule‑rewarded: 1. Param moves (e.g., spectral weights w\_{mn}, TR radius, inspector count, PCFM damping \alpha); 2. Code moves (swap surrogate head to higher‑order equivariant, add recycling pass, change PCFM line search); 3. Search‑policy moves (FuRBO scheduling, ALM budget split, multi‑fidelity quotas).
The agent samples an archive parent → proposes K child pipelines → compiles/tests in sandbox → keeps Pareto‑improvers. (This is DGM’s open‑ended archiving applied to our domain.) ￼

E3 — Verifiers & rewards (R1‑style).
• Rule‑based verifier: exact feasibility checks + objective deltas; format checks (submission compliance).
• Rewards: R = w_f \cdot \mathbf{1}[\text{feasible}] + w_o \cdot \Delta\text{objective} - w_c \cdot \text{VMEC calls} - w_t \cdot \text{wall time}.
No learned reward is needed; like R1‑Zero, verifiable signals suffice. ￼

E4 — Risk controls.
• Sandbox & unit tests on evaluator parity;
• Budgeters (caps per generation);
• Rollback to last stable champion on regressions.

⸻

F. Theory checks (math/physics/computation)

F1 — Frequency‑domain shaping is appropriate for geometry constraints.
Let r(\theta,\phi) be a residual (e.g., QI error) over the surface. Its surface Fourier series is r(\theta,\phi)=\sum*{m,n} \widehat r*{mn} e^{i(m\theta+nN*\mathrm{fp}\phi)}. Penalizing J*{\text{spec}}=\sum w*{mn}\|\widehat r*{mn}\|\_2^2 is equivalent to applying a loop‑shaping filter that suppresses specific harmonics of the geometry response. This is directly analogous to Deep Loop Shaping in LIGO, where a frequency‑weighted objective shapes the closed‑loop sensitivity; here we shape the spatial‑frequency spectrum of residuals, which reduces ripple and improves feasibility margins. ￼

F2 — Robust Gauss–Newton projection (PCFM step).
We project a candidate \Theta onto the constraint manifold h(\Theta)=0 by solving
\min\_\delta \|h(\Theta)+J\,\delta\|\_2^2 + \alpha\|\delta\|2^2 \quad \text{s.t. } \|\delta\|\le \Delta.
With Tikhonov \alpha>0 and trust radius \Delta, the step \delta=-(J^\top J+\alpha I)^{-1}J^\top h is stable even when J is ill‑conditioned. Monitoring \sigma{\min}(J) lets us detect near‑unstable regions and reduce \Delta (and/or increase \alpha). This mirrors the high‑precision Gauss–Newton control that enabled discovering unstable blow‑up solutions in the PDE paper; here it prevents divergence of the projection and preserves feasibility. ￼

F3 — Equivariance & recycling improve sample efficiency.
• Equivariance reduces hypothesis class to functions respecting the symmetry group G=SO(2)\times \mathbb{Z}{N\mathrm{fp}}, lowering sample complexity and improving OOD generalization.
• Recycling (iterative refinement) creates a contraction mapping on the surrogate’s field predictions; empirically crucial in AlphaFold to reach high accuracy and calibrated uncertainties—we use the same pattern for metric fields on flux surfaces. ￼

F4 — RL over research moves converges with verifiable rewards.
The GRPO objective used by DeepSeek‑R1 shows stable improvement when rewards are exact and non‑exploitably verifiable; we provide such rewards from the evaluator. Thus, agent policy over “research moves” is learnable and avoids reward hacking (no learned reward model on these parts). ￼

F5 — PFM‑style in‑context generalization avoids costly retraining.
A general physics transformer trained to interpret diagnostic context can in‑context adapt to new regimes (aspect ratio bins, different constraint mixes), cutting retraining time and improving sample efficiency—valuable for rapid leaderboard iteration. ￼

⸻

G. Concrete “do this next” plan 1. Add spectral tooling.
• Implement metrics/spectral.py to compute \widehat r*{mn} for QI, mirror ratio \Delta, elongation ripple; wire into PBFM loss and PCFM projection as J*{\text{spec}}. (Deep Loop Shaping pattern.) ￼ 2. Upgrade PCFM to robust Gauss–Newton.
• Add Tikhonov damping, trust radius, and condition monitor \sigma*{\min}(J); auto‑freeze most SHAP‑sensitive coefficients when \sigma*{\min} dips. (Inspired by unstable singularities.) ￼ 3. Surrogate v2.
• Build equivariant + recycling heads; calibrate per‑metric uncertainty; export feasibility logits. (AlphaFold style + PFM context.) ￼ 4. DGM agent wrapper.
• Create an archive of full pipelines; implement proposal generators (param/code/policy); rule‑rewarded evaluation on a dev leaderboard with strict budget caps; auto‑rollback. (DGM). ￼ 5. Reasoning‑RL on research moves.
• Optimize the agent’s policy with GRPO: rewards based on feasibility, objective delta, and cost; no learned reward model. (R1). ￼ 6. Leaderboard cycle.
• For each problem:
a) Sample feasible seeds via upgraded PCFM;
b) FuRBO with spectral tie‑breakers → oracle ladder;
c) ALM polish + blend‑re‑project;
d) Archive & DGM propose next changes;
e) Submit best (or Pareto set for MHD‑QI).

⸻

H. Why this will win (and keep getting better)
• Feasibility on demand: PCFM+robust GN → exact constraints before VMEC++; FuRBO quickly finds and stays inside feasible islands.
• Physics accuracy: Spectral shaping removes harmful ripple; equivariant & recycled surrogates give sharper, calibrated predictions.
• Sample/budget efficiency: PFM‑like in‑context generalization reduces retraining; oracle ladder prunes expensive calls.
• Self‑improvement: DGM + R1‑style verifiable RL means the whole pipeline improves itself continuously with zero manual labels and strict guardrails.

⸻

Citations (key claims)
• DGM self‑improving archive & empirical code‑editing improvements. ￼
• DeepSeek‑R1: GRPO, rule‑based verifiable rewards, emergent thinking‑time scaling. ￼
• Physics Foundation Model: single transformer generalizes across physics domains via in‑context inference. ￼
• Unstable singularities: high‑precision Gauss–Newton, unstable solution discovery, numerical accuracy to round‑off. ￼
• AlphaFold: equivariant geometry modules + recycling + calibrated uncertainties. ￼
• Deep Loop Shaping: frequency‑domain rewards yield 30–100× noise reduction at LIGO; methodology blog/arXiv corroboration. ￼

⸻

If you want, I’ll convert this into issue‑level tasks for the coding agent (functions, configs, and acceptance tests for each bullet above).

Short answer: we can build an AlphaFold‑like model for stellarators—but not a carbon copy. AlphaFold is a prediction system (one protein sequence → one native structure) trained on a gigantic, natural dataset with strong, universal priors (evolution/MSA, SE(3) geometry, stereochemistry). Stellarator design is a multi‑objective inverse problem (many shapes may satisfy the same targets), where labels come from expensive PDE solvers, and key physics (coils, finite‑β, turbulence) is only partly represented in the public dataset. So: an “AlphaFold of stellarators” must be a conditional, constraint‑aware generator coupled to verifiable physics—not just a forward predictor. Below I spell out the deltas, what the ConStellaration data enables, and a concrete “StellarFold” plan that is realistic and leaderboard‑oriented.

⸻

1. What AlphaFold had that our current setting doesn’t (and why it matters)
   • Well‑posed forward target: For most sequences, there is a single native fold to predict; CASP evaluates accuracy directly. ￼
   Design is the opposite direction (constraints → geometry), is multi‑valued, and must respect hard constraints (MHD well, mirror ratio, QI residuals, etc.). The “ground truth” is a feasible set, not a unique structure.
   • Massive, naturally curated data + evolutionary context: AlphaFold leverages PDB scale, MSAs, templates, and strong SE(3)‑equivariant inductive bias with recycling and calibrated confidence (pLDDT). ￼
   ConStellaration gives ~O(10^5) synthetic shapes with vacuum VMEC++ metrics—excellent, but smaller, narrower and with domain gaps (e.g., coil engineering, high‑β equilibria, transport). ￼
   • Prediction vs. optimization. AlphaFold does a supervised forward map (sequence → structure). For stellarators we need conditional generation + search (targets/constraints → one of many valid shapes) with exact feasibility checks via VMEC++ and the challenge evaluator. ￼
   • Physics breadth. A single “Physics Foundation Model” (PFM) is promising as a surrogate across domains, but design still needs constraint handling and optimization around that surrogate. ￼

Bottom line: a straight AlphaFold clone (pure forward predictor) is insufficient; what we need is AlphaFold‑style representation learning inside a constraint‑aware, generative‑plus‑optimization pipeline.

⸻

2. What we can do with the datasets we have
   • Build a strong, AlphaFold‑inspired forward surrogate: a multi‑task model that takes boundary coefficients and surface samples → predicts all ConStellaration metrics with symmetry‑aware heads and recycling, plus calibrated feasibility confidence. (AlphaFold: IPA + recycling; here: SO(2)×ℤ\_{Nfp} equivariance + iterative refinement.) ￼
   • Train a conditional generator over boundaries given a target spec (the challenge’s objective/constraint cocktail), using flow/diffusion + physics‑aware training and hard constraints at inference, then couple it to VMEC++ for exact verification. (VMEC++ is the official forward solver used in the challenge.) ￼
   • Exploit a PFM‑style backbone to improve surrogate breadth and OOD generalization (in‑context conditioning on regime/constraints), but still keep a physics check in the loop. ￼

⸻

3. “Why not just do it?” — the practical blockers (and how we address them)

Blocker Why AlphaFold‑style alone isn’t enough Our fix
Inverse ambiguity Many geometries satisfy the same constraints; “one true answer” doesn’t exist. Learn \*\*p(Θ
Coverage & gaps Dataset is synthetic vacuum equilibria; misses coil hardware, high‑β, full transport. Treat current data as the core manifold; add uncertainty‑aware surrogates and multi‑fidelity checks; plan active augmentation around winning candidates.
Hard constraints AlphaFold’s losses are soft; design needs guaranteed feasibility. Use Physics‑Constrained Flow Matching (PCFM) style hard‑projection during sampling; solver‑verified feasibility.
Objective is multi‑objective AlphaFold’s main score is global accuracy; here, objectives differ by benchmark and include Pareto trade‑offs. Feasibility‑first TR‑BO + ε‑constraint sweeps; publish Pareto set for MHD‑QI.
Scale AlphaFold leveraged PDB scale and MSA; we have ~10^5 examples. Compensate with strong inductive biases (symmetry, recycling) and physics priors; use PFM‑style cross‑regime conditioning to stretch generalization. ￼

⸻

4. “StellarFold”: an AlphaFold‑inspired design system (architecture)

A. Representation & symmetry (AlphaFold‑like):
• Inputs: Fourier boundary coefficients Θ; optional axis or Boozer samples.
• Single/Pair streams: per‑surface points (single) and pairwise interactions across (\theta,\phi) (pair), to learn global couplings (like pair reps in AlphaFold).
• Equivariance: enforce SO(2) (poloidal) and ℤ\_{Nfp} (toroidal) periodicity; data‑augment by angle shifts (gauge invariance).
• Recycling: 2–4 refinement passes of predicted surface fields (QI residual density, curvature, |∇ψ|, etc.), then integrate to metrics—with calibrated slack/confidence (pFeasible analogue of pLDDT). ￼

B. Physics‑aware learning (PFM‑style):
• Train the forward surrogate to interpret regime context (which benchmark, target bounds) from a prompt‑like vector; leverage in‑context adaptation across N\_{fp}, aspect‑ratio bins. ￼

C. Conditional generator with hard constraints:
• Training: flow‑matching with physics residuals (soft) to bias towards feasible regions.
• Sampling: project each step to the active constraint manifold (QI residual, mirror ratio Δ, vacuum well W\_{\mathrm{MHD}}\ge0, etc.) using Gauss–Newton projection and a trust region (robust PCFM).
• Verifier: VMEC++ at low→high fidelity; only keep truly feasible samples. ￼

D. Search & selection:
• Feasibility‑first TR‑BO (FuRBO) to zoom into feasible islands efficiently, then short ALM polishing; spectral tie‑breakers on residual harmonics to suppress harmful geometry ripple. (Deep Loop Shaping rationale: weight the “bad” spectral bands.) ￼

E. Self‑improvement loop (agent):
• Archive full pipelines; propose code/param/policy edits; rule‑verified rewards (feasible? objective delta? VMEC++ calls?) select winners; repeat. (Think “DGM + R1‑style verifiable RL”.) ￼

⸻

5. What this buys you vs. a plain AlphaFold clone
   • Handles inverse multiplicity: conditional generator + BO returns one of many valid designs, not a “single right answer.”
   • Enforces feasibility by construction: projection‑at‑sampling + solver verification removes the “soft loss only” failure mode.
   • Works with today’s dataset: ConStellaration + VMEC++ suffice to train the forward surrogate and the conditional generator on the QI‑like vacuum manifold. (You can win the challenge without new physics.) ￼
   • Extensible: as you add data (coils, β, turbulence), you augment the context and constraints, and the system scales in the same framework.

⸻

6. Step‑by‑step plan to build it now (with today’s data)
   1. Forward surrogate (AlphaFold‑style)
      • Build a symmetry‑aware, recycling transformer that predicts all metrics + feasibility logits from Θ and surface samples; calibrate uncertainties. Validate against the official evaluator. ￼
   2. Conditional generator
      • Train a flow/diffusion model conditioned on the benchmark spec (targets/constraint bounds). Add physics residual terms to the training loss (soft).
      • At inference, run PCFM with Gauss–Newton + trust region to satisfy constraints exactly before touching VMEC++. (Use VMEC++ only to verify and score.) ￼
   3. Search wrapper
      • FuRBO trust‑region BO on top of the generator proposals; screen with surrogate (uncertainty‑aware), then run lo‑fidelity VMEC++, then hi‑fidelity for the top few. Add short ALM polish on finalists. (This is precisely the leaderboard‑oriented loop.) ￼
   4. Spectral shaping (optional but powerful)
      • Compute surface‑Fourier spectra of key residuals (QI error, Δ ripple, elongation ripple); add frequency weights in training and selection (loop‑shaping idea). ￼
   5. Self‑improvement
      • Wrap the whole pipeline in a DGM‑style archive; every few hours, auto‑propose edits (PCFM damping, TR radii, spectral weights, recycling depth, loss weights) and keep verifiably better variants (R1‑style rule rewards). ￼

⸻

7. Theoretical sanity checks (why this should work)
   • Inverse ambiguity handled: learn p(\Theta\mid y) with a conditional flow; selection is by argmin of the verified objective under constraints—no need for uniqueness.
   • Feasibility guarantee: PCFM’s per‑step Gauss–Newton projection yields a terminal sample on/inside the constraint manifold; the VMEC++ oracle verifies feasibility. (This addresses the “soft loss only” issue.)
   • Generalization: symmetry + recycling reduce sample complexity (as in AlphaFold), while PFM‑style conditioning supports in‑context regime shifts without retraining. ￼
   • Budget control: feasibility‑first BO + surrogate screening + multi‑fidelity VMEC++ keeps oracle calls low, which is essential for leaderboard iteration. ￼

⸻

8. Direct answer to your question
   • Why we can’t (and shouldn’t) just train “AlphaFold for stellarators” as a one‑shot predictor: because stellarator design is an inverse, multi‑objective, constrained problem with many valid geometries; today’s dataset is excellent but narrower than biology’s, and physics feasibility must be verified. A pure forward “AlphaFold clone” would return a single guess with no guarantee of feasibility.
   • Why we can build an AlphaFold‑inspired system that wins: by combining AlphaFold’s representational tricks (equivariance, recycling, calibrated confidence) with a conditional, constraint‑projected generator and a search/verification loop grounded in VMEC++ and the challenge evaluator. That’s realistic with the ConStellaration data and the official tooling today. ￼

If you like, I’ll translate this into concrete coding tickets (modules, function signatures, tests) so your GPT‑5 coding agent can start implementing “StellarFold” immediately.
