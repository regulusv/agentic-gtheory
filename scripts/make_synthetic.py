"""Generate synthetic agent-evaluation data in the canonical long schema.

True random effects are drawn on the logit scale with the standard deviations
in config.yaml (synthetic.true_sigmas), step success is sampled Bernoulli, and
`reached` follows a cascade (comprehension -> tool_selection -> formatting) so
the marginal and conditional codings genuinely differ.

Run:  python scripts/make_synthetic.py [--config config.yaml]
Writes the table to paths.long_table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.build_dataset import finalize_long_table  # noqa: E402


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    sc = cfg["synthetic"]
    rng = np.random.default_rng(cfg["seed"])

    models = [f"model_{i}" for i in range(sc["n_models"])]
    domains = [f"domain_{d}" for d in range(sc["n_domains"])]
    tasks, task_domain = [], {}
    for d in domains:
        for k in range(sc["tasks_per_domain"]):
            tid = f"{d}_task_{k}"
            tasks.append(tid); task_domain[tid] = d
    skills = sc["skills"]
    judges = [f"judge_{i}" for i in range(sc["n_judges"])]
    sig = sc["true_sigmas"]

    def draw(keys, s):  # one Normal(0, s) effect per key
        return {k: rng.normal(0.0, s) for k in keys}

    e_p = draw(models, sig["p"]); e_t = draw(tasks, sig["t"]); e_s = draw(skills, sig["s"])
    e_j = draw(judges, sig["j"])
    e_pt = draw([(p, t) for p in models for t in tasks], sig["pt"])
    e_ps = draw([(p, s) for p in models for s in skills], sig["ps"])
    e_ts = draw([(t, s) for t in tasks for s in skills], sig["ts"])
    e_pts = draw([(p, t, s) for p in models for t in tasks for s in skills], sig["pts"])

    reach_prob = sc["reach_prob"]
    rows = []
    for p in models:
        for t in tasks:
            for j in judges:
                reached_prev = True
                for si, s in enumerate(skills):
                    reached = bool(reached_prev and (rng.random() < reach_prob[si]))
                    logit = (sc["grand_mean_logit"] + e_p[p] + e_t[t] + e_s[s] + e_j[j]
                             + e_pt[(p, t)] + e_ps[(p, s)] + e_ts[(t, s)] + e_pts[(p, t, s)])
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    correct = int(rng.random() < prob) if reached else 0
                    rows.append({"model": p, "task": t, "domain": task_domain[t],
                                 "skill": s, "judge": j,
                                 "reached": int(reached), "correct": correct})
                    reached_prev = reached and bool(correct)

    df = finalize_long_table(pd.DataFrame(rows))
    out = REPO_ROOT / cfg["paths"]["long_table"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[make_synthetic] wrote {len(df)} rows -> {out}")
    print(f"[make_synthetic] reached rate={df['reached'].mean():.3f} "
          f"marginal success={df['score_marginal'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    main(ap.parse_args().config)
