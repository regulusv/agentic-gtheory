"""End-to-end analysis orchestrator: long table -> G-Study -> D-Study -> figures.

Run:  python scripts/run_pipeline.py [--config config.yaml]
Assumes paths.long_table already exists (from make_synthetic.py or the real
parse+annotate+build path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import dstudy, viz                                   # noqa: E402
from src.build_dataset import coding_view                     # noqa: E402
from src.gstudy import variance_components_anova, variance_components_glmm  # noqa: E402


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = REPO_ROOT / cfg["paths"]["outputs_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.read_csv(REPO_ROOT / cfg["paths"]["long_table"])

    # ---- G-Study ---------------------------------------------------------
    gl = cfg["glmm"]
    analysis_df = coding_view(long_df, gl["coding"])
    backend = gl["backend"]
    if backend == "numpyro":
        try:
            print(f"[g-study] fitting GLMM (NumPyro), coding={gl['coding']} ...")
            vc = variance_components_glmm(
                analysis_df, num_warmup=gl["num_warmup"], num_samples=gl["num_samples"],
                num_chains=gl["num_chains"], target_accept=gl["target_accept"],
                sd_prior_scale=gl.get("sd_prior_scale", 2.0), seed=cfg["seed"],
            )
        except ImportError as e:
            print(f"[g-study] NumPyro/JAX unavailable ({e}); falling back to ANOVA baseline.")
            backend = "anova"
    if backend == "anova":
        vc = variance_components_anova(long_df)

    print("\n=== G-Study variance components "
          f"({vc.attrs.get('scale', '')}) ===")
    print(vc.to_string(index=False))

    viz.table_1_latex(vc, str(out_dir / "table_1.tex"))
    print(f"\n[table] wrote {out_dir / 'table_1.tex'}")

    # ---- Figures + D-Study ----------------------------------------------
    viz.figure_domain_variance(long_df, str(out_dir / "fig1_domain_variance.png"))
    print(f"[fig1]  wrote {out_dir / 'fig1_domain_variance.png'}")

    components = dstudy.components_from_table(vc)
    ds = cfg["dstudy"]
    viz.figure_dstudy(components, ds["n_t_grid"], ds["n_s_values"], ds["n_j_fixed"],
                      ds["phi_target"], str(out_dir / "fig2_dstudy.png"))
    print(f"[fig2]  wrote {out_dir / 'fig2_dstudy.png'}")

    tgt = ds["phi_target"]
    n_rel = dstudy.tasks_to_reach(components, tgt, n_s=3, n_j=ds["n_j_fixed"], coef="E_rho2")
    n_abs = dstudy.tasks_to_reach(components, tgt, n_s=3, n_j=ds["n_j_fixed"], coef="Phi")
    fmt = lambda n: f"{n} tasks" if n else "not reachable in range"
    print(f"\n[d-study] to reach {tgt} at n_s=3, n_j={ds['n_j_fixed']}:")
    print(f"          relative (E_rho^2, ranking)   : {fmt(n_rel)}")
    print(f"          absolute (Phi, threshold)     : {fmt(n_abs)}")
    if n_abs is None and n_rel is not None:
        print("          note: Phi is capped by a facet main effect that does not shrink with "
              "tasks\n                (e.g. single judge) -> add judges, not just tasks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    main(ap.parse_args().config)
