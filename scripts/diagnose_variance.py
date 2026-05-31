"""Variance-component diagnostics (no API calls -- pure recompute).

Reports, side by side:
  1. GLMM conditional vs marginal coding (var_mean, % of total) -- esp. t & s.
  2. Split R-hat and ESS per component at num_chains=4 (convergence check).
  3. GLMM (logit) vs balanced-ANOVA baseline (raw) -- % of total, where they diverge.

Run:  python scripts/diagnose_variance.py [--config config.yaml] [--chains 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.build_dataset import coding_view                                      # noqa: E402
from src.gstudy import variance_components_anova, variance_components_glmm      # noqa: E402

ORDER = ["p", "t", "s", "pt", "ps", "ts", "pts"]


def _fit(long_df, coding, gl, seed, chains):
    df = coding_view(long_df, coding)
    vc = variance_components_glmm(
        df, num_warmup=gl["num_warmup"], num_samples=gl["num_samples"],
        num_chains=chains, target_accept=gl["target_accept"], seed=seed,
    )
    return vc.set_index("source")


def main(config_path, chains):
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed, gl = cfg["seed"], cfg["glmm"]
    long_df = pd.read_csv(REPO_ROOT / cfg["paths"]["long_table"])
    print(f"[diag] long.csv rows={len(long_df)} | chains={chains} "
          f"| warmup={gl['num_warmup']} samples={gl['num_samples']}\n")

    print("[diag] fitting GLMM (conditional) ...")
    cond = _fit(long_df, "conditional", gl, seed, chains)
    print("[diag] fitting GLMM (marginal) ...")
    marg = _fit(long_df, "marginal", gl, seed, chains)
    print("[diag] ANOVA baseline (balanced p x t x s, score_marginal) ...")
    anova = variance_components_anova(long_df).set_index("source")

    def g(tbl, src, col):
        return tbl.loc[src, col] if src in tbl.index else float("nan")

    # ---- 1. conditional vs marginal -------------------------------------
    print("\n" + "=" * 78)
    print("1) GLMM variance components: CONDITIONAL vs MARGINAL (logit scale)")
    print("=" * 78)
    print(f"{'src':>4} | {'cond var':>9} {'cond %':>7} | {'marg var':>9} {'marg %':>7} "
          f"| {'Δ%(m-c)':>8}")
    print("-" * 78)
    for s in ORDER:
        cv, cp = g(cond, s, "var_mean"), g(cond, s, "pct_of_total")
        mv, mp = g(marg, s, "var_mean"), g(marg, s, "pct_of_total")
        print(f"{s:>4} | {cv:>9.3f} {cp:>6.1f}% | {mv:>9.3f} {mp:>6.1f}% | {mp-cp:>+7.1f}%")

    # ---- 2. R-hat / ESS at chains=4 -------------------------------------
    print("\n" + "=" * 78)
    print(f"2) Convergence at num_chains={chains}: split R-hat (<1.01 good) and ESS")
    print("=" * 78)
    print(f"{'src':>4} | {'cond Rhat':>9} {'cond ESS':>9} | {'marg Rhat':>9} {'marg ESS':>9}")
    print("-" * 78)
    worst = 0.0
    for s in ORDER:
        cr, ce = g(cond, s, "r_hat"), g(cond, s, "ess")
        mr, me = g(marg, s, "r_hat"), g(marg, s, "ess")
        flag = " <-- check" if (cr == cr and cr >= 1.01) or (mr == mr and mr >= 1.01) else ""
        worst = max(worst, cr if cr == cr else 0, mr if mr == mr else 0)
        print(f"{s:>4} | {cr:>9.3f} {ce:>9.0f} | {mr:>9.3f} {me:>9.0f}{flag}")
    print(f"\n  worst R-hat across both codings = {worst:.3f} "
          f"({'all converged' if worst < 1.01 else 'SOME NOT CONVERGED'})")

    # ---- 3. GLMM vs ANOVA (% of total) ----------------------------------
    print("\n" + "=" * 78)
    print("3) % of total variance: GLMM (logit) vs ANOVA baseline (raw, continuous)")
    print("   scales differ -> compare shares/ordering, not absolute magnitudes")
    print("=" * 78)
    print(f"{'src':>4} | {'GLMM-cond %':>11} {'GLMM-marg %':>11} {'ANOVA %':>9} | {'note':>20}")
    print("-" * 78)
    for s in ORDER:
        cp, mp, ap = g(cond, s, "pct_of_total"), g(marg, s, "pct_of_total"), g(anova, s, "pct_of_total")
        note = ""
        if ap == ap and mp == mp and abs(ap - mp) >= 10:
            note = "LARGE divergence"
        print(f"{s:>4} | {cp:>10.1f}% {mp:>10.1f}% {ap:>8.1f}% | {note:>20}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()
    main(a.config, a.chains)
