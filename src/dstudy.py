"""D-Study: forecast benchmark reliability from G-Study variance components.

Object of measurement = Model (p). For a hypothetical benchmark sampling
n_t tasks, n_s skill steps and n_j judges per model:

    sigma^2_rel = var_pt/n_t + var_ps/n_s + var_pts/(n_t*n_s)
    sigma^2_abs = sigma^2_rel + var_t/n_t + var_s/n_s + var_j/n_j
                                + var_ts/(n_t*n_s)
    E_rho^2 (relative, ranking)   = var_p / (var_p + sigma^2_rel)
    Phi     (absolute, threshold) = var_p / (var_p + sigma^2_abs)

Only effects present in the components dict contribute; missing ones are 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Divisor for each effect = product of the sampled facet sizes it averages over.
# (Effects involving p are NOT divided by n_p, since p is the object of measurement.)
_DIVISORS = {
    "p": lambda nt, ns, nj: 1.0,
    "t": lambda nt, ns, nj: nt,
    "s": lambda nt, ns, nj: ns,
    "j": lambda nt, ns, nj: nj,
    "pt": lambda nt, ns, nj: nt,
    "ps": lambda nt, ns, nj: ns,
    "ts": lambda nt, ns, nj: nt * ns,
    "pts": lambda nt, ns, nj: nt * ns,
}
_REL_EFFECTS = {"pt", "ps", "pts"}                 # involve p
_ABS_EXTRA = {"t", "s", "j", "ts"}                 # facet-only effects


def coefficients(components: dict, n_t: int, n_s: int, n_j: int) -> dict:
    var_p = components.get("p", 0.0)
    rel = sum(components.get(e, 0.0) / _DIVISORS[e](n_t, n_s, n_j) for e in _REL_EFFECTS)
    abs_ = rel + sum(components.get(e, 0.0) / _DIVISORS[e](n_t, n_s, n_j) for e in _ABS_EXTRA)
    return {
        "n_t": n_t, "n_s": n_s, "n_j": n_j,
        "E_rho2": var_p / (var_p + rel) if (var_p + rel) > 0 else np.nan,
        "Phi": var_p / (var_p + abs_) if (var_p + abs_) > 0 else np.nan,
    }


def sweep(components: dict, n_t_grid, n_s_values, n_j_fixed: int = 1) -> pd.DataFrame:
    """Reliability over a grid of (n_t, n_s) at fixed n_j."""
    rows = [coefficients(components, nt, ns, n_j_fixed)
            for ns in n_s_values for nt in n_t_grid]
    return pd.DataFrame(rows)


def tasks_to_reach(components: dict, target: float, n_s: int, n_j: int,
                   coef: str = "Phi", max_tasks: int = 5000) -> int | None:
    """Smallest integer n_t such that the chosen coefficient >= target.

    coef: "Phi" (absolute/threshold decisions) or "E_rho2" (relative/ranking).
    Returns None if the target is not reached within max_tasks -- which, for
    Phi, often means a facet main effect that does not shrink with n_t (e.g. a
    single judge, n_j=1) is capping absolute dependability.
    """
    for nt in range(1, max_tasks + 1):
        if coefficients(components, nt, n_s, n_j)[coef] >= target:
            return nt
    return None


def components_from_table(vc: pd.DataFrame) -> dict:
    """Extract a {source: variance} dict from a gstudy output table."""
    col = "var_mean" if "var_mean" in vc.columns else "var"
    return dict(zip(vc["source"], vc[col]))
