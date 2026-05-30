# CLAUDE.md — project context for Claude Code

This file is auto-loaded by Claude Code. It captures the design decisions made
so far so you can continue without re-deriving them.

## What this project is

A course project (CS321M, AI Measurement Science) applying **Generalizability
Theory** to agentic LLM tool-use evaluation. We decompose the variance in
ToolBench-style benchmark scores across four facets — Model (p), Task (t),
Skill (s), Judge (j) — and run a Decision Study to recommend how many
tasks/skills/judges a benchmark needs for a target dependability (Φ > 0.8).
The accompanying paper draft is the source of truth for framing; key decisions:

- **Skills** = {comprehension, tool_selection, formatting}, treated as a
  multivariate/fixed dimension conceptually (we want per-skill reliability, not
  to average over skills). The univariate p×t×s decomposition is reported for
  comparability.
- **Sequential dependency**: do NOT naively code post-failure steps as 0 only.
  We carry two codings — `marginal` (absorbing 0) and `conditional` (hurdle:
  score only steps actually reached). Conditional is primary.
- **Estimator**: a crossed-random-effects **logistic GLMM** (NumPyro), not
  balanced ANOVA — because the outcome is binary (ANOVA homoscedasticity is
  violated) and trajectories are unbalanced. Variance components are on the
  logit scale; there is no explicit residual (Bernoulli variance is implicit).
- **Judge** is a real facet: a multi-judge sub-study quantifies grader variance.

## Canonical data schema (one row per cell)

`model, task, domain, skill, judge, reached(0/1), correct(0/1), score_marginal(0/1)`

Defined in `src/build_dataset.py` (`SKILLS`, `COLUMNS`, `finalize_long_table`).
Both the synthetic generator and the real path MUST emit exactly this schema so
`gstudy`/`dstudy`/`viz` are identical for both.

## File map

- `src/build_dataset.py` — schema, `coding_view(df, coding)`, real-path join.
- `src/gstudy.py` — `variance_components_glmm` (primary), `variance_components_anova` (baseline).
- `src/dstudy.py` — `coefficients`, `sweep`, `tasks_to_reach`, `components_from_table`.
- `src/viz.py` — `figure_domain_variance`, `figure_dstudy`, `table_1_latex`.
- `scripts/make_synthetic.py` — synthetic data (quickstart).
- `scripts/run_pipeline.py` — orchestrator.
- `src/parse_toolbench.py`, `src/annotate_judge.py` — REAL PATH STUBS to implement.

## Conventions

- All randomness seeded from `config.yaml:seed`. Don't hardcode seeds elsewhere.
- Keep the synthetic and real paths producing the identical long schema.
- Run the synthetic quickstart (`bash scripts/run_all.sh`) after any change to
  `gstudy`/`dstudy`/`viz` to confirm the pipeline still goes green.
- New deps → add to `requirements.txt`; never commit API keys or the judge cache.

## TODO (priority order for the handoff)

1. **Download data**: get `reproduction_data.zip` from the ToolBench Google Drive
   (linked in the ToolBench README), unzip to `data/raw/reproduction_data/`.
   Then `python scripts/build_real.py --dry-run` to confirm parsing + coverage
   with NO API calls. `src/parse_toolbench.py` already targets this structure
   (verified offline on mock data); if real JSON differs, adapt `_iter_nodes`
   and `_skill_reached` (print one JSON first) — these are the only spots tied
   to the file schema.
2. **Judge (OpenAI, already implemented)** — `src/annotate_judge.py` uses the
   OpenAI client (`_call_judge`: retry/backoff + robust VERDICT parsing + atomic
   incremental cache). Set `export OPENAI_API_KEY=...`, add judge id(s) to
   `real.judges` (e.g. `gpt-4o`), then smoke-test before scaling:
   `python -m src.annotate_judge --smoke 5 --judge gpt-4o`
   (re-running should make 0 new calls = cache works). Then full run:
   `python scripts/build_real.py` -> long.csv -> `python scripts/run_pipeline.py`.
3. **Multi-judge sub-study** — add >=2 ids to `real.judges`; confirm the GLMM's
   judge-variance estimate is sane against Cohen's κ.
4. **ANOVA baseline rigor** — the EMS estimator needs a balanced p×t×s array;
   real data is unbalanced, so prefer the GLMM there (ANOVA is baseline/Fig1).
5. **Conference-extension hooks** (later): treat inference `method` (DFS vs CoT)
   as a real facet; cross-benchmark replication (τ-bench / BFCL); Bayesian
   hierarchical handling of variable-length trajectories; external-validity
   check predicting leaderboard rank-stability from the D-Study.

## Data sourcing (resolved)

The evaluated models (`p` facet) come from ToolBench `reproduction_data/`, which
ships pre-computed trajectories for multiple model×method dirs (chatgpt_cot,
chatgpt_dfs, gpt4_dfs, toolllama_dfs, ...). **No model APIs are needed** — this
is secondary analysis. The ONLY API key required is for the judge (`j` facet).
`config.yaml:real.method_filter='dfs'` keeps one method so p×t stays clean;
`task` = `subset::query_id` (e.g. `G1_instruction::1010`). It is subset-qualified
because ToolBench query_ids are assigned per test-subset and DO collide across
subsets (~13 in chatgpt_dfs); a bare qid would conflate distinct queries (and two
domains) into one task. `subset::qid` is still shared across model dirs, so the
p×t design stays crossed. NB: `real.raw_dir` points at
`reproduction_data/model_predictions/` (the per-query trajectories live one level
below the unzip root; the sibling `model_predictions_converted/` is an aggregated
format we do not use).

## Sanity expectations on synthetic data

The synthetic `true_sigmas` give a real-but-modest Model variance and large
Task / Model×Task variance. On the quickstart you should see:
- `p`, `pt`, `ps`, `ts` recovered close to truth (these facets have many levels);
- `s` (3 levels) and `j` (3 levels) with inflated point estimates and **very wide
  credible intervals** — the expected "few-level facets are imprecise" behavior
  the paper flags, not a bug;
- task-related variance (t + pt + ts + pts) dominating the Model component;
- relative reliability (E_rho^2) reaching the target with enough tasks, while
  absolute dependability (Phi) is **capped by the single-judge term** (a main
  effect that does not shrink with more tasks) — the empirical argument for a
  multi-judge protocol.

Validate the estimator's correctness on a *no-cascade* synthetic (set all
`reach_prob` to 1.0): then marginal == conditional and recovery is cleanest,
since the conditional coding's selection-on-the-cascade is removed.
