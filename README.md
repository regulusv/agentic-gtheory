# Deconstructing Agentic Capabilities — G-Theory Tool-Use Evaluation

Variance decomposition and benchmark-design analysis for agentic LLM tool-use
evaluation, using Generalizability Theory (G-Theory). Estimates how much of the
variance in agent benchmark scores is attributable to the Model, Task, Skill,
and Judge facets, then runs a Decision Study (D-Study) to forecast how many
tasks/skills/judges a benchmark needs to reach a target reliability.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
```

## Quickstart (synthetic data, no API calls)

Reproduces the full pipeline end-to-end on synthetic data with a known
variance structure — useful to verify the install and the analysis code
before any real annotation:

```bash
bash scripts/run_all.sh
```

Outputs land in `outputs/`:
- `table_1.tex` — variance components (logit-scale GLMM, with 95% credible intervals)
- `fig1_domain_variance.png` — per-domain variance decomposition
- `fig2_dstudy.png` — dependability Φ vs number of tasks, for n_s ∈ {1, 3, 5}

Expected runtime: under ~3 minutes on a laptop CPU (NUTS sampling dominates).

## Repository structure

```
config.yaml              All seeds, paths, model/task lists, GLMM & D-study settings
src/
  parse_toolbench.py     [real] raw ToolBench logs -> trajectory records      (stub)
  annotate_judge.py      [real] LLM-as-judge grading + JSON cache             (stub)
  build_dataset.py       canonical long-format schema + marginal/conditional codings
  gstudy.py              GLMM (NumPyro, primary) + ANOVA-EMS baseline
  dstudy.py              E-rho^2 / Phi reliability sweeps
  viz.py                 Figure 1, Figure 2, Table 1
scripts/
  make_synthetic.py      synthetic data generator (quickstart)
  build_real.py          reproduction_data -> long.csv (parse -> judge -> build)
  run_pipeline.py        long table -> G-Study -> D-Study -> figures
  run_all.sh             synthetic + pipeline (quickstart)
```

## Reproducing the real analysis

1. Download `reproduction_data.zip` from the ToolBench repo's Google Drive,
   unzip it to `data/raw/reproduction_data/`. These are pre-computed trajectories
   for multiple model×method directories (e.g. `chatgpt_dfs`, `gpt4_dfs`,
   `toolllama_dfs`) — the evaluated models (`p` facet). **No model APIs needed.**
2. Validate parsing with no API calls:
   ```bash
   python scripts/build_real.py --dry-run
   ```
3. Add judge id(s) to `real.judges` in `config.yaml` and implement the
   `_call_judge` provider call in `src/annotate_judge.py` (needs your API key —
   the only key the project requires). Then:
   ```bash
   python scripts/build_real.py      # parse -> judge (cached) -> long.csv
   python scripts/run_pipeline.py    # G-Study -> D-Study -> figures + Table 1
   ```

`config.yaml:real.method_filter` keeps a single inference method (default `dfs`)
so the model×task design stays clean; set it to `null` to model `method` as an
extra facet. Which script produces which paper artifact is printed to stdout.

## Notes

- The GLMM reads variance components on the **logit scale**; there is no
  separate residual term (Bernoulli sampling variance is the implicit residual).
- The ANOVA-EMS estimator treats the outcome as continuous and is included only
  as the **baseline** the paper critiques, plus for the per-domain Figure 1.
- Set the outcome coding (`conditional` vs `marginal`) in `config.yaml`.
