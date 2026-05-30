"""Parse ToolBench `reproduction_data` into per-step trajectory records.

DATA SOURCE
-----------
Download `reproduction_data.zip` from the ToolBench repo's Google Drive, unzip,
and point `raw_dir` at the resulting `reproduction_data/` folder. Its layout is:

    reproduction_data/
      <model>_<method>/            # one dir per (model x method) -> our `p` facet
        G1_instruction/            # six test subsets (G1/G2/G3 x inst/cat/tool)
          10160_CoT@1.json         # one JSON per query = one trajectory
          ...
        G1_category/  G1_tool/  G2_instruction/  G2_category/  G3_instruction/

Examples of <model>_<method> dirs: chatgpt_cot, chatgpt_dfs, gpt4_dfs,
toolllama_dfs, ... Each directory is a distinct evaluated model/method whose
trajectories already exist -- so NO model API calls are needed (secondary
analysis). Only the LLM judge (src/annotate_judge.py) needs an API key.

WHAT THIS PRODUCES
------------------
One row per (trajectory, skill) with the *structural* `reached` flag:
    trajectory_id, model, task, domain, skill, reached(0/1)
The binary correctness label per (trajectory, skill) is added later by the
judge (annotate_judge) and merged in build_dataset.build_long_table.

DESIGN DECISIONS (see CLAUDE.md / paper)
----------------------------------------
* `model`  := the directory name (e.g. "chatgpt_dfs"). To keep p x t clean we
  recommend fixing ONE method (default: DFS dirs) via `method_filter`; treating
  method as its own facet is a documented future extension.
* `task`   := the query_id (stable across models -> gives a crossed p x t).
* `domain` := mapped from the test subset (G1/G2/G3 * inst/cat/tool). ToolBench
  is organised by tool *category*, not by semantic domain like "Finance"; we use
  the subset as the coarse domain grouping for Figure 1. (TODO: optionally remap
  to RapidAPI top-level categories using toolenv metadata for finer domains.)
* `reached` per skill, derived from the trajectory structure:
    - comprehension : always reached (the model received the query).
    - tool_selection: reached iff the trajectory contains >=1 tool/API call node.
    - formatting    : reached iff >=1 tool call actually executed (got a response).
  These are deliberately simple, auditable heuristics over the answer-tree /
  conversation nodes; tune `_skill_reached` to the exact JSON you observe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_dataset import SKILLS

# Six official ToolBench test subsets, mapped to a coarse "domain" for Figure 1.
TEST_SUBSETS = [
    "G1_instruction", "G1_category", "G1_tool",
    "G2_instruction", "G2_category", "G3_instruction",
]
SUBSET_TO_DOMAIN = {s: s for s in TEST_SUBSETS}  # identity; remap if desired

# Substrings used to recognise method from a directory name.
DFS_TOKENS = ("dfs", "dfsdt", "dfs_wofilter")
COT_TOKENS = ("cot", "react")


def _detect_method(model_dir_name: str) -> str:
    n = model_dir_name.lower()
    if any(tok in n for tok in DFS_TOKENS):
        return "dfs"
    if any(tok in n for tok in COT_TOKENS):
        return "cot"
    return "other"


def _iter_nodes(traj: dict):
    """Yield step/node dicts from one trajectory JSON, robust to schema variants.

    ToolBench answers appear in a few shapes across versions:
      * {"answer_generation": {"train_messages"/"chain": [...]}}
      * {"answer": {...}} or a list of conversation turns with "role"/"node".
    We walk whatever list of message/node dicts we can find. ADAPT to the exact
    JSON you see after unzipping (print one file first).
    """
    if not isinstance(traj, dict):
        return
    ag = traj.get("answer_generation", traj)
    for key in ("train_messages", "chain", "messages", "conversation", "nodes"):
        seq = ag.get(key) if isinstance(ag, dict) else None
        if isinstance(seq, list):
            for item in seq:
                # train_messages may be a list-of-lists of message dicts.
                if isinstance(item, list):
                    yield from (m for m in item if isinstance(m, dict))
                elif isinstance(item, dict):
                    yield item
            return


def _has_tool_call(nodes: list[dict]) -> bool:
    for m in nodes:
        if m.get("role") in ("tool", "function", "assistant"):
            if m.get("function_call") or m.get("tool_calls") or m.get("name"):
                return True
        if "Action" in (m.get("node_type", ""), m.get("type", "")):
            return True
    return False


def _has_tool_response(nodes: list[dict]) -> bool:
    for m in nodes:
        if m.get("role") in ("tool", "function", "observation"):
            return True
        if m.get("node_type") in ("Action Input", "Observation") or "observation" in m:
            return True
    return False


def _skill_reached(skill: str, nodes: list[dict]) -> int:
    if skill == "comprehension":
        return 1
    if skill == "tool_selection":
        return int(_has_tool_call(nodes))
    if skill == "formatting":
        return int(_has_tool_response(nodes))
    raise ValueError(f"unknown skill {skill!r}")


def _query_id_from(traj: dict, fallback_stem: str) -> str:
    for k in ("query_id", "id", "qid"):
        if isinstance(traj, dict) and traj.get(k) is not None:
            return str(traj[k])
    # filenames look like "10160_CoT@1.json" -> take the leading numeric id.
    return fallback_stem.split("_")[0]


def _extract_query(traj: dict, nodes: list[dict]) -> str:
    """Best-effort recovery of the user query/instruction text."""
    for k in ("query", "instruction", "Goal", "goal"):
        v = traj.get(k) if isinstance(traj, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    for m in nodes:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    return ""


def _trajectory_text(nodes: list[dict], max_chars: int = 4000) -> str:
    """Compact, judge-readable rendering of the trajectory's reasoning + actions.

    Keeps roles/types and content; truncates to `max_chars` to bound token cost.
    Adapt the field names if your JSON labels reasoning/actions differently.
    """
    parts = []
    for m in nodes:
        role = m.get("role") or m.get("node_type") or m.get("type") or "step"
        content = m.get("content")
        if content is None and m.get("function_call"):
            fc = m["function_call"]
            content = f"call {fc.get('name', '?')}({fc.get('arguments', '')})"
        if content is None and m.get("name"):
            content = f"tool={m['name']}"
        if content:
            parts.append(f"[{role}] {str(content)[:600]}")
    text = "\n".join(parts)
    return text[:max_chars]


def parse_toolbench(raw_dir: str, method_filter: str | None = "dfs",
                    models: list[str] | None = None) -> pd.DataFrame:
    """Walk `reproduction_data/` and emit (trajectory, skill) rows.

    Parameters
    ----------
    raw_dir : path to the unzipped `reproduction_data/` directory.
    method_filter : keep only dirs whose detected method == this ("dfs" or "cot").
        None keeps all (then method should be modelled as a facet).
    models : optional explicit allow-list of <model>_<method> dir names. If given,
        overrides method_filter.
    """
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download reproduction_data.zip from the ToolBench "
            f"Google Drive, unzip, and point real.raw_dir at the reproduction_data/ "
            f"folder. To exercise the pipeline meanwhile, run scripts/make_synthetic.py."
        )

    model_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if models:
        model_dirs = [d for d in model_dirs if d.name in set(models)]
    elif method_filter:
        model_dirs = [d for d in model_dirs if _detect_method(d.name) == method_filter]
    if not model_dirs:
        raise ValueError(f"No model dirs selected under {root} "
                         f"(method_filter={method_filter!r}, models={models!r}).")

    rows = []
    for mdir in model_dirs:
        model = mdir.name
        for subset in TEST_SUBSETS:
            sdir = mdir / subset
            if not sdir.is_dir():
                continue
            domain = SUBSET_TO_DOMAIN[subset]
            for jf in sorted(sdir.glob("*.json")):
                try:
                    traj = json.loads(jf.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                nodes = list(_iter_nodes(traj))
                qid = _query_id_from(traj, jf.stem)
                tid = f"{model}::{subset}::{qid}"
                query = _extract_query(traj, nodes)
                traj_text = _trajectory_text(nodes)
                for skill in SKILLS:
                    rows.append({
                        "trajectory_id": tid, "model": model, "task": qid,
                        "domain": domain, "skill": skill,
                        "reached": _skill_reached(skill, nodes),
                        # Carried into the judge prompt (annotate_judge.context_cols):
                        "query": query, "trajectory_text": traj_text,
                    })

    if not rows:
        raise ValueError(f"Parsed 0 trajectories under {root}. Inspect one JSON "
                         f"file and adapt _iter_nodes/_skill_reached to its schema.")
    df = pd.DataFrame(rows)
    # Keep only tasks present for >=2 models so the p x t design is (near-)crossed.
    counts = df.groupby("task")["model"].nunique()
    crossed = counts[counts >= 2].index
    df = df[df["task"].isin(crossed)].reset_index(drop=True)
    return df
