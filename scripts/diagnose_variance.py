"""Variance-component diagnostics (no API calls -- pure recompute).

Reports, side by side, across BOTH RE-SD priors (primary + sensitivity):
  1. GLMM components (var_mean, % of total) for conditional & marginal coding,
     under each prior -- so pts's prior-sensitivity is reported explicitly.
  2. Split R-hat and ESS per component (convergence check) under each prior.
  3. GLMM (logit, primary prior) vs balanced-ANOVA baseline (raw) -- % of total.

Primary prior = config glmm.sd_prior_scale; the sensitivity prior defaults to
2.0 (weakly-informative) and can be overridden with --sensitivity-prior.

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
CODINGS = ["conditional", "marginal"]


def _fit(long_df, coding, gl, seed, chains, prior):
    df = coding_view(long_df, coding)
    return variance_components_glmm(
        df, num_warmup=gl["num_warmup"], num_samples=gl["num_samples"],
        num_chains=chains, target_accept=gl["target_accept"],
        sd_prior_scale=prior, seed=seed,
    ).set_index("source")


def main(config_path, chains, sensitivity_prior):
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed, gl = cfg["seed"], cfg["glmm"]
    primary = gl.get("sd_prior_scale", 1.0)
    priors = [primary] + ([sensitivity_prior] if sensitivity_prior != primary else [])
    long_df = pd.read_csv(REPO_ROOT / cfg["paths"]["long_table"])
    print(f"[diag] long.csv rows={len(long_df)} | chains={chains} "
          f"| warmup={gl['num_warmup']} samples={gl['num_samples']} ta={gl['target_accept']}")
    print(f"[diag] priors: primary HalfNormal({primary}) | sensitivity HalfNormal({sensitivity_prior})\n")

    fits = {}  # (prior, coding) -> table
    for pr in priors:
        for cd in CODINGS:
            print(f"[diag] fitting GLMM  prior=HN({pr})  coding={cd} ...")
            fits[(pr, cd)] = _fit(long_df, cd, gl, seed, chains, pr)
    print("[diag] ANOVA baseline (balanced p x t x s, score_marginal) ...")
    anova = variance_components_anova(long_df).set_index("source")

    def g(tbl, src, col):
        return tbl.loc[src, col] if (tbl is not None and src in tbl.index) else float("nan")

    # ---- 1. components (% of total) across prior x coding ----------------
    print("\n" + "=" * 92)
    print("1) GLMM % of total variance  (PRIOR-SENSITIVITY: watch pts move with the prior)")
    print("=" * 92)
    hdr = f"{'src':>4} |"
    for pr in priors:
        for cd in CODINGS:
            hdr += f" HN{pr}/{cd[:4]:>4}"
    print(hdr); print("-" * 92)
    for s in ORDER:
        line = f"{s:>4} |"
        for pr in priors:
            for cd in CODINGS:
                line += f" {g(fits[(pr, cd)], s, 'pct_of_total'):>9.1f}"
        print(line)

    # ---- 2. R-hat / ESS across prior x coding ----------------------------
    print("\n" + "=" * 92)
    print(f"2) Convergence at chains={chains}: R-hat (<1.01 good) / ESS")
    print("=" * 92)
    for pr in priors:
        print(f"\n  prior = HalfNormal({pr})")
        print(f"  {'src':>4} | {'cond Rhat':>9} {'cond ESS':>9} | {'marg Rhat':>9} {'marg ESS':>9}")
        print("  " + "-" * 56)
        for s in ORDER:
            cr, ce = g(fits[(pr, 'conditional')], s, 'r_hat'), g(fits[(pr, 'conditional')], s, 'ess')
            mr, me = g(fits[(pr, 'marginal')], s, 'r_hat'), g(fits[(pr, 'marginal')], s, 'ess')
            flag = " <-- not converged" if (cr == cr and cr >= 1.01) or (mr == mr and mr >= 1.01) else ""
            print(f"  {s:>4} | {cr:>9.3f} {ce:>9.0f} | {mr:>9.3f} {me:>9.0f}{flag}")

    # ---- 3. GLMM (primary prior) vs ANOVA --------------------------------
    print("\n" + "=" * 92)
    print(f"3) % of total: GLMM primary HN({primary}) vs ANOVA baseline (raw, continuous)")
    print("   scales differ -> compare shares/ordering, not absolute magnitudes")
    print("=" * 92)
    print(f"{'src':>4} | {'cond %':>8} {'marg %':>8} {'ANOVA %':>8} | note")
    print("-" * 92)
    for s in ORDER:
        cp = g(fits[(primary, 'conditional')], s, 'pct_of_total')
        mp = g(fits[(primary, 'marginal')], s, 'pct_of_total')
        ap = g(anova, s, 'pct_of_total')
        note = "LARGE divergence" if (ap == ap and mp == mp and abs(ap - mp) >= 10) else ""
        print(f"{s:>4} | {cp:>7.1f}% {mp:>7.1f}% {ap:>7.1f}% | {note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--sensitivity-prior", type=float, default=2.0,
                    help="second RE-SD prior scale to report for sensitivity")
    a = ap.parse_args()
    main(a.config, a.chains, a.sensitivity_prior)
