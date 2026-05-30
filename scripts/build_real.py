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


def main(config_path: str, dry_run: bool) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    rc = cfg["real"]
    raw_dir = REPO_ROOT / rc["raw_dir"]

    trajectories = parse_toolbench(
        str(raw_dir),
        method_filter=rc.get("method_filter"),
        models=rc.get("models") or None,
    )
    n_traj = trajectories["trajectory_id"].nunique()
    print(f"[parse] {n_traj} trajectories | models={sorted(trajectories['model'].unique())}")
    print(f"[parse] tasks={trajectories['task'].nunique()} "
          f"| reached rate per skill:")
    print(trajectories.groupby('skill', observed=True)['reached'].mean().to_string())

    if dry_run:
        print("\n[dry-run] parsing OK; skipping judge annotation and CSV write.")
        return

    judges = rc.get("judges") or []
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
                    help="parse trajectories and report coverage without judge calls")
    a = ap.parse_args()
    main(a.config, a.dry_run)
