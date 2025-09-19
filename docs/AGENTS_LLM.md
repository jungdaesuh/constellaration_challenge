# LLM Portfolio & Roles (Optional)

Scope
- Default ConStelX runs are deterministic; LLMs are optional.
- When enabled, LLMs should interact only by emitting ablation specs for `constelx ablate run` and writing brief analyses into run artifacts. All final decisions and scores must come from VMEC++/official metrics.

Recommended models by role
- Planner / Orchestrator (deep reasoning)
  - GPT‑5 (OpenAI): primary planner/reviewer when budget allows.
  - o3 / o3‑mini (OpenAI): strong deliberate reasoning; cost‑effective inner loops.
  - Gemini 2.5 Pro (Google DeepMind): large context, multi‑modal planning.
  - Claude Sonnet 4 / Opus 4 (Anthropic): robust long‑form reasoning/coding.
  - Grok‑3 family (xAI): diversity check and ensemble vote.

- Coding Agent (tool use, refactors, debugging)
  - GPT‑5; Claude Opus 4 / Sonnet 4; Gemini 2.5 Pro.
  - Open‑weights options: DeepSeek‑V3 (MoE), Qwen2.5‑Coder, StarCoder2.

- Small open‑weights (local utility, privacy)
  - Phi‑4 / Phi‑4‑reasoning (Microsoft).
  - Llama 3.2 / Code Llama (Meta).

Usage pattern
- Keep the optimization loop deterministic. Use an LLM “planner” to generate an ablation suite (e.g., change penalty schedules, PCFM damping/iters, surrogate features), run it via `constelx ablate run`, then apply only the empirically best patch.
- Gate LLM outputs with novelty/sanity checks and always VMEC++‑verify before archiving/submitting.

Notes
- Record prompts, model versions, and seeds alongside artifacts for reproducibility.
- Prefer on‑device small models for documentation and minor code suggestions.
