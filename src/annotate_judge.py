"""LLM-as-a-judge annotation of agent trajectories (OpenAI), with on-disk caching.

REAL PATH -- run locally with your own OpenAI API key (the chat sandbox cannot
reach external LLM APIs). Verdicts are cached to JSON keyed by
(trajectory_id, skill, judge) so you never re-pay for an annotation, and the
cache is flushed incrementally so a crash mid-run keeps everything paid for so far.

Output DataFrame columns (consumed by src.build_dataset.build_long_table):
    trajectory_id, skill, judge, correct (0/1)

Setup:
    pip install openai
    export OPENAI_API_KEY="sk-..."      # never hard-code or commit the key

Design:
  * One grading call per (trajectory, skill, judge) cell.
  * The cache is the single source of truth; calls are skipped on cache hit.
  * `_call_judge` does the OpenAI call + robust verdict parse + retry/backoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Per-skill rubric. Keep these verbatim in paper Appendix A for reproducibility.
RUBRICS = {
    "comprehension": "Did the agent correctly parse the user's intent before acting?",
    "tool_selection": "Did the agent select an appropriate API/tool from the available pool for this intent?",
    "formatting": "Was the agent's tool/API call a syntactically valid, well-formed payload (valid JSON/arguments)?",
}

SYSTEM_MSG = (
    "You are a strict evaluator of LLM tool-use agent trajectories. "
    "You grade ONE criterion at a time and return a binary verdict. "
    "Reason briefly, then output a final line of exactly 'VERDICT: 1' (criterion met) "
    "or 'VERDICT: 0' (criterion not met)."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*([01])", re.IGNORECASE)


def _cache_key(trajectory_id: str, skill: str, judge: str) -> str:
    return hashlib.sha1(f"{trajectory_id}|{skill}|{judge}".encode()).hexdigest()


def _build_prompt(step_context: dict, skill: str) -> str:
    """Assemble the grading prompt for one (trajectory, skill) cell.

    `step_context` should carry enough of the trajectory for the judge to grade:
    at minimum the user query and the agent's relevant action(s). The parser
    currently emits only structural fields, so populate `trajectory_text` (and
    optionally `query`, `available_tools`) when building the annotation frame --
    see annotate()'s `context_cols`. Unknown/missing fields are simply omitted.
    """
    ctx = {k: v for k, v in step_context.items()
           if k in ("query", "available_tools", "trajectory_text", "task", "model")
           and v not in (None, "")}
    return (
        f"Criterion to grade -- {skill}: {RUBRICS[skill]}\n\n"
        f"--- agent trajectory (context) ---\n"
        f"{json.dumps(ctx, ensure_ascii=False, indent=2)}\n"
        f"--- end context ---\n\n"
        f"Grade ONLY the '{skill}' criterion. Reason briefly, then end with "
        f"a line exactly 'VERDICT: 1' or 'VERDICT: 0'."
    )


_CLIENT = None


def _client():
    """Lazily create a module-level OpenAI client (reused across calls)."""
    global _CLIENT
    if _CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai to run judge annotation.") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in your environment before annotating.")
        _CLIENT = OpenAI()
    return _CLIENT


def _parse_verdict(text: str) -> int:
    """Extract the binary verdict; prefer the explicit VERDICT line, else last 0/1."""
    if not text:
        raise ValueError("empty judge response")
    matches = _VERDICT_RE.findall(text)
    if matches:
        return int(matches[-1])
    tail = re.findall(r"[01]", text.strip())
    if tail:
        return int(tail[-1])
    raise ValueError(f"could not parse a 0/1 verdict from: {text[:200]!r}")


def _call_judge(prompt: str, judge: str, temperature: float = 0.0, seed: int = 0,
                max_retries: int = 4) -> int:
    """OpenAI chat call -> binary verdict, with exponential backoff on errors."""
    client = _client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge,
                temperature=temperature,
                seed=seed,
                messages=[{"role": "system", "content": SYSTEM_MSG},
                          {"role": "user", "content": prompt}],
            )
            return _parse_verdict(resp.choices[0].message.content)
        except Exception as e:  # network / rate-limit / parse error
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"judge call failed after {max_retries} retries: {last_err}")


def annotate(trajectories: pd.DataFrame, judges: list[str], cache_path: str,
             temperature: float = 0.0, seed: int = 0,
             context_cols: tuple[str, ...] = ("query", "available_tools", "trajectory_text"),
             flush_every: int = 25) -> pd.DataFrame:
    """Annotate every reached (trajectory, skill) for each judge, caching to disk.

    `context_cols` are the optional columns carrying trajectory text/query into
    the prompt; include them in `trajectories` for higher-quality grading.
    The cache is flushed every `flush_every` new calls and at the end.
    """
    cache_file = Path(cache_path)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2))
        tmp.replace(cache_file)  # atomic

    rows, n_calls = [], 0
    for _, r in trajectories.iterrows():
        if not r["reached"]:
            continue  # only grade steps actually reached (hurdle)
        ctx = {c: r[c] for c in context_cols if c in r.index}
        ctx.update({"task": r.get("task"), "model": r.get("model")})
        for j in judges:
            key = _cache_key(r["trajectory_id"], r["skill"], j)
            if key not in cache:
                cache[key] = _call_judge(_build_prompt(ctx, r["skill"]), j,
                                         temperature=temperature, seed=seed)
                n_calls += 1
                if n_calls % flush_every == 0:
                    flush()
            rows.append({"trajectory_id": r["trajectory_id"], "skill": r["skill"],
                         "judge": j, "correct": int(cache[key])})

    flush()
    print(f"[annotate_judge] {n_calls} new calls; {len(cache)} cached verdicts -> {cache_file}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Smoke test: grade the first N reached cells with one judge, then exit.
#   python -m src.annotate_judge --smoke 5 --judge gpt-4o
# Requires OPENAI_API_KEY and a parsed reproduction_data/ (uses config.yaml).
# --------------------------------------------------------------------------
def _smoke(n: int, judge: str, config_path: str) -> None:
    import yaml
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from src.parse_toolbench import parse_toolbench
    cfg = yaml.safe_load((Path(config_path)).read_text())
    rc = cfg["real"]
    traj = parse_toolbench(str(repo / rc["raw_dir"]),
                           method_filter=rc.get("method_filter"),
                           models=rc.get("models") or None)
    reached = traj[traj["reached"] == 1].head(n).copy()
    print(f"[smoke] grading {len(reached)} reached cells with judge={judge} ...")
    cache = repo / cfg["paths"]["processed_dir"] / "smoke.judgecache.json"
    out = annotate(reached, [judge], str(cache), temperature=rc.get("judge_temperature", 0.0))
    print(out.to_string(index=False))
    print(f"[smoke] re-run should make 0 new calls (cache hit). Cache: {cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=5, help="number of cells to grade")
    ap.add_argument("--judge", default="gpt-4o", help="OpenAI model id for the judge")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    _smoke(a.smoke, a.judge, a.config)
