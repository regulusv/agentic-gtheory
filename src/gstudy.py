"""G-Study: estimate variance components of a crossed-random-effects design
over Models (p), Tasks (t), Skills (s), Judges (j).

Two estimators:
  * variance_components_glmm  -- PRIMARY. Bayesian logistic mixed model
      (NumPyro/NUTS). Reads variance components on the LOGIT scale and returns
      posterior mean + 95% credible interval for each. Appropriate for binary
      outcomes (no homoscedasticity assumption) and unbalanced data.
  * variance_components_anova -- BASELINE. Classical balanced-ANOVA expected-
      mean-squares estimator on a p x t x s array (judges averaged out, outcome
      treated as continuous). This is the "naive" estimator we critique; kept
      for comparison and used per-domain in Figure 1.

The GLMM linear predictor matches Eq. (1) in the paper:
    logit P(Y=1) = mu + nu_p + nu_t + nu_s + nu_j
                      + nu_pt + nu_ps + nu_ts + nu_pts
On the logit scale there is no separate residual variance parameter; the
Bernoulli sampling variance plays that role.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

# Effects modelled, in table order. Tuples name the facets in each interaction.
EFFECTS = {
    "p": ("model",),
    "t": ("task",),
    "s": ("skill",),
    "j": ("judge",),
    "pt": ("model", "task"),
    "ps": ("model", "skill"),
    "ts": ("task", "skill"),
    "pts": ("model", "task", "skill"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _codes(df: pd.DataFrame, cols) -> tuple[np.ndarray, int]:
    """Integer-encode a (possibly composite) grouping; return codes and #levels."""
    key = df[list(cols)].astype(str).agg("||".join, axis=1)
    cats = pd.Categorical(key)
    return cats.codes.astype(np.int64), len(cats.categories)


# ---------------------------------------------------------------------------
# PRIMARY: Bayesian GLMM (NumPyro)
# ---------------------------------------------------------------------------
def variance_components_glmm(
    df: pd.DataFrame,
    effects=None,
    num_warmup: int = 400,
    num_samples: int = 400,
    num_chains: int = 1,
    target_accept: float = 0.9,
    sd_prior_scale: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit the crossed-RE logistic GLMM and summarise variance components.

    `df` must contain the facet columns and a binary outcome column `y`
    (produced by build_dataset.coding_view).
    `sd_prior_scale` is the scale of the HalfNormal prior on every random-effect
    SD (logit scale). 2.0 is weakly-informative; tighten (e.g. 1.0) to regularise
    the heavy tail of near-separated interaction variances (pts) and aid mixing.
    Returns a DataFrame: [source, var_mean, ci_low, ci_high, pct_of_total, r_hat, ess].
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(num_chains)

    if effects is None:
        # Drop any effect whose facet has <2 observed levels (e.g. single judge).
        effects = {k: v for k, v in EFFECTS.items()
                   if all(df[c].nunique() >= 2 for c in v)}

    code_arrays, n_levels = {}, {}
    for name, cols in effects.items():
        code_arrays[name], n_levels[name] = _codes(df, cols)
    y = df["y"].to_numpy().astype(np.float32)
    idx = {k: jnp.asarray(v) for k, v in code_arrays.items()}
    yj = jnp.asarray(y)

    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
        logits = mu
        for name in effects:
            sd = numpyro.sample(f"sd_{name}", dist.HalfNormal(sd_prior_scale))
            # Non-centred parameterisation for stable sampling.
            with numpyro.plate(f"plate_{name}", n_levels[name]):
                z = numpyro.sample(f"z_{name}", dist.Normal(0.0, 1.0))
            numpyro.deterministic(f"var_{name}", sd ** 2)
            logits = logits + (sd * z)[idx[name]]
        numpyro.sample("y", dist.Bernoulli(logits=logits), obs=yj)

    mcmc = MCMC(
        NUTS(model, target_accept_prob=target_accept),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    post = mcmc.get_samples()

    # Convergence diagnostics (split R-hat, ESS) per variance component. R-hat
    # needs >=2 chains; with 1 chain it is reported as NaN.
    from numpyro.diagnostics import summary as _summary
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = _summary({f"var_{n}": grouped[f"var_{n}"] for n in effects},
                    group_by_chain=True)

    rows = []
    total = np.sum([np.mean(np.asarray(post[f"var_{n}"])) for n in effects])
    for name in effects:
        v = np.asarray(post[f"var_{name}"])
        d = diag[f"var_{name}"]
        rows.append({
            "source": name,
            "var_mean": float(v.mean()),
            "ci_low": float(np.percentile(v, 2.5)),
            "ci_high": float(np.percentile(v, 97.5)),
            "pct_of_total": 100.0 * float(v.mean()) / total,
            "r_hat": float(d.get("r_hat", np.nan)),
            "ess": float(d.get("n_eff", np.nan)),
        })
    out = pd.DataFrame(rows)
    out.attrs["scale"] = "logit"
    out.attrs["note"] = ("Logit-scale variance components. No explicit residual: "
                         "Bernoulli sampling variance is the implicit residual.")
    return out


# ---------------------------------------------------------------------------
# BASELINE: classical balanced-ANOVA EMS on a p x t x s array
# ---------------------------------------------------------------------------
def variance_components_anova(df: pd.DataFrame, value: str = "score_marginal") -> pd.DataFrame:
    """Expected-mean-squares variance components for a balanced, fully-crossed
    p x t x s random-effects design (one value per cell; judges averaged out).

    This treats the outcome as continuous -- the estimator the paper critiques
    for binary data -- and is provided only as a baseline / for per-domain
    Figure 1. Negative estimates are truncated to 0 (standard practice).
    """
    cell = (df.groupby(["model", "task", "skill"], observed=True)[value]
              .mean().reset_index())
    models = sorted(cell["model"].unique())
    tasks = sorted(cell["task"].unique())
    skills = list(pd.Categorical(cell["skill"]).categories)
    np_, nt, ns = len(models), len(tasks), len(skills)
    if min(np_, nt, ns) < 2:
        raise ValueError("ANOVA EMS needs >=2 levels on each of model/task/skill.")

    X = (cell.pivot_table(index="model", columns=["task", "skill"], values=value, observed=True)
             .reindex(index=models))
    A = X.to_numpy().reshape(np_, nt, ns)  # [p, t, s]
    if np.isnan(A).any():
        raise ValueError("ANOVA EMS requires a fully balanced p x t x s array (no missing cells).")

    g = A.mean()
    mp = A.mean(axis=(1, 2)); mt = A.mean(axis=(0, 2)); ms = A.mean(axis=(0, 1))
    mpt = A.mean(axis=2); mps = A.mean(axis=1); mts = A.mean(axis=0)

    SS_p = nt * ns * np.sum((mp - g) ** 2)
    SS_t = np_ * ns * np.sum((mt - g) ** 2)
    SS_s = np_ * nt * np.sum((ms - g) ** 2)
    SS_pt = ns * np.sum((mpt - mp[:, None] - mt[None, :] + g) ** 2)
    SS_ps = nt * np.sum((mps - mp[:, None] - ms[None, :] + g) ** 2)
    SS_ts = np_ * np.sum((mts - mt[:, None] - ms[None, :] + g) ** 2)
    pred = (mpt[:, :, None] + mps[:, None, :] + mts[None, :, :]
            - mp[:, None, None] - mt[None, :, None] - ms[None, None, :] + g)
    SS_pts = np.sum((A - pred) ** 2)

    def ms_(ss, df_):
        return ss / df_ if df_ > 0 else 0.0

    MS_p, MS_t, MS_s = ms_(SS_p, np_ - 1), ms_(SS_t, nt - 1), ms_(SS_s, ns - 1)
    MS_pt = ms_(SS_pt, (np_ - 1) * (nt - 1))
    MS_ps = ms_(SS_ps, (np_ - 1) * (ns - 1))
    MS_ts = ms_(SS_ts, (nt - 1) * (ns - 1))
    MS_pts = ms_(SS_pts, (np_ - 1) * (nt - 1) * (ns - 1))

    var = {
        "pts": MS_pts,
        "pt": (MS_pt - MS_pts) / ns,
        "ps": (MS_ps - MS_pts) / nt,
        "ts": (MS_ts - MS_pts) / np_,
        "p": (MS_p - MS_pt - MS_ps + MS_pts) / (nt * ns),
        "t": (MS_t - MS_pt - MS_ts + MS_pts) / (np_ * ns),
        "s": (MS_s - MS_ps - MS_ts + MS_pts) / (np_ * nt),
    }
    var = {k: max(0.0, v) for k, v in var.items()}  # truncate negatives
    total = sum(var.values()) or 1.0
    order = ["p", "t", "s", "pt", "ps", "ts", "pts"]
    out = pd.DataFrame([{"source": k, "var": var[k], "pct_of_total": 100 * var[k] / total}
                        for k in order])
    out.attrs["scale"] = "raw (continuous ANOVA baseline)"
    return out
