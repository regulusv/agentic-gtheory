"""Regenerate Table 1 + fig2 from the committed PRIMARY GLMM spec, plus a
prior-sensitivity column from the converged HN(2.0) fit. No API calls.

fig1 (ANOVA baseline) is prior-independent and is intentionally NOT touched.

Outputs:
  outputs/table_1.tex             primary HN(sd_prior_scale) with CI + R-hat + ESS
  outputs/table_1_sensitivity.tex sensitivity HN(2.0), same sampler
  outputs/fig2_dstudy.png         D-Study from the primary components

Run:  python scripts/regenerate_outputs.py
"""
from __future__ import annotations

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


def main():
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    gl, ds, seed = cfg["glmm"], cfg["dstudy"], cfg["seed"]
    out = REPO_ROOT / cfg["paths"]["outputs_dir"]
    long_df = pd.read_csv(REPO_ROOT / cfg["paths"]["long_table"])
    df = coding_view(long_df, gl["coding"])
    primary_scale = gl.get("sd_prior_scale", 1.0)

    def fit(prior):
        return variance_components_glmm(
            df, num_warmup=gl["num_warmup"], num_samples=gl["num_samples"],
            num_chains=gl["num_chains"], target_accept=gl["target_accept"],
            sd_prior_scale=prior, seed=seed,
        ).set_index("source").reindex(ORDER).reset_index()

    print(f"[regen] coding={gl['coding']} sampler={gl['num_warmup']}/{gl['num_samples']}"
          f"x{gl['num_chains']}@{gl['target_accept']}")
    print(f"[regen] fitting PRIMARY HalfNormal({primary_scale}) ...")
    primary = fit(primary_scale)
    print(f"[regen] fitting SENSITIVITY HalfNormal(2.0) ...")
    sens = fit(2.0)

    viz.table_1_latex(primary, str(out / "table_1.tex"))
    viz.table_1_latex(sens, str(out / "table_1_sensitivity.tex"))
    comp = dstudy.components_from_table(primary)
    viz.figure_dstudy(comp, ds["n_t_grid"], ds["n_s_values"], ds["n_j_fixed"],
                      ds["phi_target"], str(out / "fig2_dstudy.png"))

    def show(vc, label):
        print(f"\n=== Table 1 :: {label} (logit scale, conditional) ===")
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
        p20 = dstudy.coefficients(c, 20, 3, 1)["Phi"]
        p100 = dstudy.coefficients(c, 100, 3, 1)["Phi"]
        print(f"  {label:18}: E_rho2 @20={e20:.3f} @100={e100:.3f} asymptote={einf:.3f} "
              f"| Phi @20={p20:.3f} @100={p100:.3f}")
    print(f"\n[regen] wrote {out/'table_1.tex'}, {out/'table_1_sensitivity.tex'}, {out/'fig2_dstudy.png'}")
    print("[regen] fig1_domain_variance.png intentionally unchanged (prior-independent ANOVA).")


if __name__ == "__main__":
    main()
