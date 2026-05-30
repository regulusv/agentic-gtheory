"""Assemble the canonical tidy long-format table consumed by the rest of the
pipeline.

CANONICAL SCHEMA (one row per (model, task, skill, judge) cell)
---------------------------------------------------------------
    model     : str   -- model identifier (object of measurement, "p")
    task      : str   -- task identifier ("t")
    domain    : str   -- task domain (Finance / Weather / ...), for Figure 1
    skill     : str   -- one of SKILLS ("s"); ordered comprehension->...->formatting
    judge     : str   -- judge identifier ("j")
    reached   : int   -- 1 if the trajectory reached this skill step, else 0
    correct   : int   -- 1 if the step was performed correctly (valid iff reached==1)
    score_marginal : int -- correct if reached else 0  (absorbing/marginal coding)

Both `make_synthetic.py` (synthetic path) and `build_long_table` (real path)
emit exactly this schema, so gstudy/dstudy/viz are identical for both.
"""
from __future__ import annotations

import pandas as pd

SKILLS = ["comprehension", "tool_selection", "formatting"]
COLUMNS = [
    "model", "task", "domain", "skill", "judge",
    "reached", "correct", "score_marginal",
]


def finalize_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, derive `score_marginal`, and order columns."""
    missing = {"model", "task", "domain", "skill", "judge", "reached", "correct"} - set(df.columns)
    if missing:
        raise ValueError(f"long table missing required columns: {sorted(missing)}")
    df = df.copy()
    df["reached"] = df["reached"].astype(int)
    # `correct` is only meaningful where reached==1; coerce unreached to 0 for storage.
    df.loc[df["reached"] == 0, "correct"] = 0
    df["correct"] = df["correct"].astype(int)
    df["score_marginal"] = (df["reached"] & df["correct"]).astype(int)
    df["skill"] = pd.Categorical(df["skill"], categories=SKILLS, ordered=True)
    return df[COLUMNS].sort_values(["model", "task", "skill", "judge"]).reset_index(drop=True)


def build_long_table(trajectories: pd.DataFrame, judge_scores: pd.DataFrame) -> pd.DataFrame:
    """REAL PATH. Join parsed trajectories with cached judge scores into the
    canonical long table.

    Parameters
    ----------
    trajectories : DataFrame from src.parse_toolbench
        Expected columns: [trajectory_id, model, task, domain, skill, reached]
    judge_scores : DataFrame from src.annotate_judge
        Expected columns: [trajectory_id, skill, judge, correct]

    TODO(real-data): the join keys below assume one (skill) row per trajectory
    step. If a trajectory issues multiple tool calls, decide how to collapse
    repeated comprehension/tool_selection moments into one binary score per
    skill (see paper Limitations: construct validity of the skill taxonomy).
    """
    merged = trajectories.merge(
        judge_scores, on=["trajectory_id", "skill"], how="left", validate="many_to_many"
    )
    if merged["correct"].isna().any():
        n = int(merged["correct"].isna().sum())
        raise ValueError(f"{n} (trajectory, skill) cells have no judge score; check annotation coverage.")
    return finalize_long_table(merged)


def coding_view(df: pd.DataFrame, coding: str) -> pd.DataFrame:
    """Return the analysis frame for a given outcome coding.

    "marginal"    -> every cell, outcome = score_marginal (post-failure -> 0).
    "conditional" -> only reached cells, outcome = correct (hurdle; primary).
    """
    df = df.copy()
    if coding == "marginal":
        df["y"] = df["score_marginal"].astype(int)
        return df
    if coding == "conditional":
        df = df[df["reached"] == 1].copy()
        df["y"] = df["correct"].astype(int)
        return df
    raise ValueError(f"unknown coding {coding!r}; use 'marginal' or 'conditional'")
