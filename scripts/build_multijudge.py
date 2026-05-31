"""Multi-judge sub-study: annotate a stratified task subset with extra judges.

The primary judge (real.judges, e.g. gpt-4o) already grades the full balanced
sample (scripts/build_real.py). Here we add real.judges_multi (e.g.
claude-sonnet + gpt-4o-mini) on a stratified `multi_judge_tasks`-task subset, so
every reached (model, task, skill) cell in the subset is graded by ALL judges.
That replication identifies the 3-way (pts) interaction (otherwise confounded
with residual under a single judge) and makes judge variance estimable.

Reuses the incremental judge cache keyed by (trajectory_id, skill, judge), so an
interrupted run resumes without re-paying. gpt-* -> OpenAI, claude-* -> Anthropic.

Run (dry):   python scripts/build_multijudge.py --dry-run
Run (real):  python scripts/build_multijudge.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_real import stratified_balanced_sample, estimate_cost              # noqa: E402
from src.annotate_judge import annotate                                       # noqa: E402
from src.build_dataset import build_long_table                                # noqa: E402
from src.parse_toolbench import parse_toolbench                               # noqa: E402


def pick_task_subset(traj: pd.DataFrame, n_tasks: int, seed: int) -> set:
    """Stratified (by domain/subset) seeded subset of `n_tasks` tasks."""
    task_dom = traj.drop_duplicates("task")[["task", "domain"]]
    per = n_tasks // task_dom["domain"].nunique()
    picked = [grp.sample(n=min(per, len(grp)), random_state=seed)["task"]
              for _, grp in task_dom.groupby("domain")]
    keep = list(pd.concat(picked))
    if len(keep) < n_tasks:  # top up to exactly n_tasks
        extra = task_dom[~task_dom["task"].isin(keep)].sample(
            n=n_tasks - len(keep), random_state=seed)["task"]
        keep += list(extra)
    return set(keep[:n_tasks])


def main(config_path: str, dry_run: bool) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    rc, seed = cfg["real"], cfg["seed"]

    full = parse_toolbench(str(REPO_ROOT / rc["raw_dir"]),
                           method_filter=rc.get("method_filter"),
                           models=rc.get("models") or None)
    sample = stratified_balanced_sample(full, int(rc["sample"]["tasks_per_subset"]), seed)
    subset_tasks = pick_task_subset(sample, int(rc["multi_judge_tasks"]), seed)
    sub = sample[sample["task"].isin(subset_tasks)].reset_index(drop=True)

    all_judges = list(rc["judges"]) + list(rc["judges_multi"])
    print(f"[multi] subset tasks={sub['task'].nunique()} "
          f"reached cells={int(sub['reached'].sum())}")
    print(f"[multi] all judges={all_judges} (primary cached; extras new)")
    # New calls = extras x reached cells (primary judge already cached).
    est = estimate_cost(sub, rc["judges_multi"])
    print(f"[multi] expected NEW calls (extras only): "
          f"{est['calls']} (~{est['reached_cells']} cells x {est['n_judges']} extra judges)")
    for j in rc["judges_multi"]:
        ej = estimate_cost(sub, [j])
        print(f"        {j:24} {ej['reached_cells']} calls  ~${ej['usd']:.2f}")

    if dry_run:
        print("\n[dry-run] no annotation; no spend.")
        return

    cache = REPO_ROOT / cfg["paths"]["processed_dir"] / "judge_cache.judgecache.json"
    scores = annotate(sub, all_judges, str(cache),
                      temperature=rc.get("judge_temperature", 0.0))
    long_df = build_long_table(sub, scores)
    out = REPO_ROOT / cfg["paths"]["processed_dir"] / "long_multijudge.csv"
    long_df.to_csv(out, index=False)
    print(f"\n[multi] wrote {len(long_df)} rows ({long_df['judge'].nunique()} judges) -> {out}")
    print("[multi] judges present:", sorted(long_df["judge"].unique()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report subset size + expected calls/cost without spending")
    a = ap.parse_args()
    main(a.config, a.dry_run)
