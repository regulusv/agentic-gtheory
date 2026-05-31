"""Regenerate Table 1 + fig2 from the committed PRIMARY GLMM spec, plus a
prior-sensitivity column from the converged HN(2.0) fit. No API calls.

fig1 (ANOVA baseline) is prior-independent and is intentionally NOT touched.

Per the CLAUDE.md provenance convention, every artifact filename encodes its
spec (prior x samples x chains) and every file embeds the git commit + seed, so
paper numbers can never be silently mixed across runs/commits. Sampler defaults
come from config.yaml:glmm but can be overridden (e.g. --samples 4000) to push
the main effects under R-hat<1.01 for the headline table.

Run:  python scripts/regenerate_outputs.py [--warmup N --samples N --chains N]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import dstudy, viz                                   # noqa: E402
from src.build_dataset import coding_view                     # noqa: E402
from src.gstudy import variance_components_glmm               # noqa: E402

ORDER = ["p", "t", "s", "pt", "ps", "ts", "pts"]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _stamp(path: Path, provenance: str) -> None:
    """Prepend a provenance comment line to a written .tex file."""
    txt = path.read_text(encoding="utf-8")
    path.write_text(f"% PROVENANCE: {provenance}\n{txt}", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    gl, ds, seed = cfg["glmm"], cfg["dstudy"], cfg["seed"]
    ap.add_argument("--warmup", type=int, default=gl["num_warmup"])
    ap.add_argument("--samples", type=int, default=gl["num_samples"])
    ap.add_argument("--chains", type=int, default=gl["num_chains"])
    a = ap.parse_args()

    out = REPO_ROOT / cfg["paths"]["outputs_dir"]
    long_df = pd.read_csv(REPO_ROOT / cfg["paths"]["long_table"])
    df = coding_view(long_df, gl["coding"])
    primary_scale = gl.get("sd_prior_scale", 1.0)
    commit = _git_hash()

    def spec(prior):
        return f"HN{prior}_{a.samples}x{a.chains}"

    def provenance(prior):
        return (f"commit={commit} seed={seed} coding={gl['coding']} "
                f"prior=HalfNormal({prior}) sampler={a.warmup}w/{a.samples}s"
                f"x{a.chains}c@{gl['target_accept']}")

    def fit(prior):
        return variance_components_glmm(
            df, num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
            target_accept=gl["target_accept"], sd_prior_scale=prior, seed=seed,
        ).set_index("source").reindex(ORDER).reset_index()

    print(f"[regen] coding={gl['coding']} sampler={a.warmup}/{a.samples}x{a.chains}"
          f"@{gl['target_accept']} commit={commit} seed={seed}")
    print(f"[regen] fitting PRIMARY HalfNormal({primary_scale}) ...")
    primary = fit(primary_scale)
    print(f"[regen] fitting SENSITIVITY HalfNormal(2.0) ...")
    sens = fit(2.0)

    f_primary = out / f"table_1_{spec(primary_scale)}.tex"
    f_sens = out / f"table_1_{spec(2.0)}.tex"
    f_fig2 = out / f"fig2_dstudy_{spec(primary_scale)}.png"
    viz.table_1_latex(primary, str(f_primary)); _stamp(f_primary, provenance(primary_scale))
    viz.table_1_latex(sens, str(f_sens)); _stamp(f_sens, provenance(2.0))
    comp = dstudy.components_from_table(primary)
    viz.figure_dstudy(comp, ds["n_t_grid"], ds["n_s_values"], ds["n_j_fixed"],
                      ds["phi_target"], str(f_fig2))

    def show(vc, label):
        print(f"\n=== Table 1 :: {label} (logit, conditional) ===")
        print(f"{'src':>4} {'sigma^2':>9} {'%tot':>7} {'95% CI':>22} {'R-hat':>7} {'ESS':>7}")
        for _, r in vc.iterrows():
            ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            print(f"{r['source']:>4} {r['var_mean']:>9.3f} {r['pct_of_total']:>6.1f}% "
                  f"{ci:>22} {r['r_hat']:>7.3f} {r['ess']:>7.0f}")
        print(f"  worst R-hat = {vc['r_hat'].max():.3f}")

    show(primary, f"PRIMARY HN({primary_scale})")
    show(sens, "SENSITIVITY HN(2.0)")

    print("\n=== D-Study (n_s=3, n_j=1): E_rho^2 ===")
    for label, vc in [(f"PRIMARY HN({primary_scale})", primary), ("SENS HN(2.0)", sens)]:
        c = dstudy.components_from_table(vc)
        e20 = dstudy.coefficients(c, 20, 3, 1)["E_rho2"]
        e100 = dstudy.coefficients(c, 100, 3, 1)["E_rho2"]
        einf = dstudy.coefficients(c, 1_000_000, 3, 1)["E_rho2"]
        print(f"  {label:18}: E_rho2 @20={e20:.3f} @100={e100:.3f} asymptote={einf:.3f}")

    print(f"\n[regen] wrote {f_primary.name}, {f_sens.name}, {f_fig2.name}")
    print("[regen] fig1_domain_variance.png intentionally unchanged (prior-independent ANOVA).")


if __name__ == "__main__":
    main()
