"""Real-data builder: reproduction_data/ -> canonical long.csv.

Pipeline:
  1. parse_toolbench(raw_dir)         -> (trajectory, skill, reached) rows
  2. annotate(trajectories, judges)   -> (trajectory, skill, judge, correct)   [needs API key]
  3. build_long_table(...)            -> canonical long schema
  4. write to paths.long_table

Run (real):     python scripts/build_real.py --config config.yaml
Run (dry):      python scripts/build_real.py --config config.yaml --dry-run
The dry run parses trajectories and prints coverage WITHOUT any judge API calls,
so you can validate parsing before spending anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.annotate_judge import annotate                       # noqa: E402
from src.build_dataset import build_long_table                # noqa: E402
from src.parse_toolbench import parse_toolbench                # noqa: E402

# Rough per-call token assumptions for the cost estimate (one judge grading one
# (trajectory, skill) cell). The prompt carries the rubric + query + a
# <=4000-char trajectory_text; the judge replies with brief reasoning + VERDICT.
# Tune these / the price table to your provider's real numbers.
_TOK_IN_PER_CALL = 1500
_TOK_OUT_PER_CALL = 150
# USD per 1M tokens (input, output). Update to current pricing.
_PRICE = {
    "gpt-4o":      (2.50, 10.00),
    "gpt-4o-mini": (0.15,  0.60),
}


def stratified_balanced_sample(traj: pd.DataFrame, n_per_subset: int,
                               seed: int) -> pd.DataFrame:
    """Keep a balanced, fully-crossed p x t sub-matrix.

    1. Restrict to tasks present for ALL observed models (so every (model, task)
       cell exists -> p x t is full; with 3 skills the p x t x s array is
       complete and the balanced-ANOVA baseline is well-defined).
    2. Within each subset/domain, randomly sample up to `n_per_subset` of those
       fully-covered tasks. Seeded for reproducibility (config.yaml:seed).
    """
    n_models = traj["model"].nunique()
    cov = traj.groupby("task")["model"].nunique()
    full = cov[cov == n_models].index
    sub = traj[traj["task"].isin(full)]
    # one (task -> domain) row; tasks are subset-qualified so domain is unique.
    task_dom = sub.drop_duplicates("task")[["task", "domain"]]
    picked = []
    for domain, grp in task_dom.groupby("domain"):
        take = min(n_per_subset, len(grp))
        picked.append(grp.sample(n=take, random_state=seed)["task"])
    keep = pd.concat(picked) if picked else pd.Series([], dtype=object)
    return traj[traj["task"].isin(set(keep))].reset_index(drop=True)


def estimate_cost(traj: pd.DataFrame, judges: list[str]) -> dict:
    """Judge calls + USD estimate. One call per reached (trajectory, skill, judge)."""
    reached_cells = int(traj["reached"].sum())          # reached (traj, skill) cells
    n_judges = max(1, len(judges))
    calls = reached_cells * n_judges
    tok_in = calls * _TOK_IN_PER_CALL
    tok_out = calls * _TOK_OUT_PER_CALL
    # Price by the first judge id if known, else gpt-4o as a default.
    pin, pout = _PRICE.get(judges[0] if judges else "gpt-4o", _PRICE["gpt-4o"])
    usd = tok_in / 1e6 * pin + tok_out / 1e6 * pout
    return {"reached_cells": reached_cells, "n_judges": n_judges, "calls": calls,
            "tok_in": tok_in, "tok_out": tok_out, "usd": usd}


def _print_coverage(traj: pd.DataFrame, label: str) -> None:
    cov = (traj.groupby("skill", observed=True)["reached"]
           .agg(reached="sum", total="size"))
    cov["rate"] = cov["reached"] / cov["total"].clip(lower=1)
    print(f"[{label}] trajectories={traj['trajectory_id'].nunique()} "
          f"tasks={traj['task'].nunique()} models={traj['model'].nunique()}")
    print(f"[{label}] reached coverage per skill:")
    for skill, row in cov.iterrows():
        print(f"    {skill:14} {int(row['reached']):>7}/{int(row['total']):<7} "
              f"= {row['rate']:.4f}")


def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def main(config_path: str, dry_run: bool, no_sample: bool = False) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    rc = cfg["real"]
    seed = cfg["seed"]
    raw_dir = REPO_ROOT / rc["raw_dir"]
    judges = rc.get("judges") or []

    full = parse_toolbench(
        str(raw_dir),
        method_filter=rc.get("method_filter"),
        models=rc.get("models") or None,
    )
    print(f"[parse] models={sorted(full['model'].unique())}")
    _print_coverage(full, "full")

    # ---- stratified balanced sub-matrix -------------------------------------
    sc = rc.get("sample") or {}
    use_sample = bool(sc.get("enabled", False)) and not no_sample
    trajectories = full
    if use_sample:
        n_per = int(sc.get("tasks_per_subset", 40))
        trajectories = stratified_balanced_sample(full, n_per, seed)
        print(f"\n[sample] stratified balanced: <={n_per} fully-covered tasks/subset, "
              f"seed={seed}")
        _print_coverage(trajectories, "sample")

    # ---- cost estimate: full vs sampled -------------------------------------
    price_id = judges[0] if judges else "gpt-4o"
    ef, es = estimate_cost(full, judges or [price_id]), estimate_cost(trajectories, judges or [price_id])
    print(f"\n[cost] one call per reached (trajectory, skill, judge); "
          f"judges={judges or [price_id]} (priced as {price_id})")
    print(f"[cost] assume {_TOK_IN_PER_CALL} in + {_TOK_OUT_PER_CALL} out tokens/call")
    print(f"    {'':8} {'reached':>9} {'calls':>9} {'tok_in':>12} {'tok_out':>10} {'est_USD':>10}")
    for lbl, e in (("full", ef), ("sample", es)):
        print(f"    {lbl:8} {e['reached_cells']:>9} {e['calls']:>9} "
              f"{e['tok_in']:>12,} {e['tok_out']:>10,} {_fmt_usd(e['usd']):>10}")
    if use_sample:
        print(f"    sampling cuts judge calls to "
              f"{100*es['calls']/max(1,ef['calls']):.1f}% of full.")

    if dry_run:
        print("\n[dry-run] parsing OK; skipping judge annotation and CSV write.")
        return

    if not judges:
        raise SystemExit("real.judges is empty in config.yaml -- add >=1 judge id "
                         "and implement annotate_judge._call_judge before a real run.")
    cache = REPO_ROOT / cfg["paths"]["processed_dir"] / "judge_cache.judgecache.json"
    judge_scores = annotate(trajectories, judges, str(cache),
                            temperature=rc.get("judge_temperature", 0.0))

    long_df = build_long_table(trajectories, judge_scores)
    out = REPO_ROOT / cfg["paths"]["long_table"]
    out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out, index=False)
    print(f"\n[build] wrote {len(long_df)} rows -> {out}")
    print("[build] now run: python scripts/run_pipeline.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="parse trajectories and report coverage/cost without judge calls")
    ap.add_argument("--no-sample", action="store_true",
                    help="ignore real.sample and annotate the full corpus")
    a = ap.parse_args()
    main(a.config, a.dry_run, a.no_sample)
