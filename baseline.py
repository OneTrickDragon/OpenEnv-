"""
baseline.py — Baseline inference script for the Data Cleaning OpenEnv.

Runs a prompted LLM (default: gpt-4o) against all three tasks using the
real OpenEnv client interface and reports reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

    # Against a live HF Space:
    python baseline.py --env-url https://YOUR-USERNAME-data-cleaning-openenv.hf.space

    # Single task:
    python baseline.py --task ecommerce_easy --seed 42

    # Quiet mode (scores only):
    python baseline.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Stub openenv-core if not installed (allows syntax check without package) ──
try:
    from openenv.core.env_client import EnvClient  # noqa: F401
except ImportError:
    import types as _t
    for _name in ["openenv","openenv.core","openenv.core.env_server",
                  "openenv.core.env_client","openenv.core.client_types"]:
        sys.modules.setdefault(_name, _t.ModuleType(_name))

from client import DataCleaningEnv
from models import DataCleaningAction

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a data engineering expert inside a Python REPL.
    You have a messy pandas DataFrame called `df` and must clean it.

    Rules:
    - Write Python code that modifies `df` in place.
    - Available: pd, np, re, math, datetime, difflib, collections (all pre-injected).
    - No file I/O. No network. Max 20 lines per response.
    - Write ONLY Python code — no explanation, no markdown fences.
    - When finished, respond with exactly: SUBMIT
""").strip()


def build_user_message(obs, step: int, prev_result: str = "") -> str:
    parts = [
        f"TASK:\n{obs.task_spec}",
        f"DATAFRAME (first 10 rows):\n{obs.df_preview}",
        f"DTYPES:\n{obs.df_info}",
        f"PARTIAL SCORE: {obs.partial_score:.3f}  |  STEP: {step}/5",
    ]
    if prev_result:
        parts.append(f"LAST OUTPUT:\n{prev_result[:300]}")
    if obs.error:
        parts.append(f"ERROR: {obs.error[:200]}")
    parts.append("Your code (or SUBMIT):")
    return "\n\n".join(parts)


def parse_response(text: str) -> tuple[str, bool]:
    text = text.strip()
    if text.upper().startswith("SUBMIT") or not text:
        return "", True
    # Strip accidental markdown fences
    text = re.sub(r"```(?:python)?\n?", "", text).strip("`").strip()
    return text, False

def run_episode(
    env_url:  str,
    llm,
    model:    str,
    task_id:  str,
    seed:     int,
    verbose:  bool = True,
) -> dict:
    """Run one full episode with the prompted LLM agent."""
    if verbose:
        print(f"\n{'='*60}\n  Task: {task_id}   Seed: {seed}   Model: {model}\n{'='*60}")

    start = time.time()

    with DataCleaningEnv(base_url=env_url).sync() as env:
        result = env.reset(task_id=task_id, seed=seed)
        obs = result.observation

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(obs, step=0)},
        ]

        final_reward = 0.0

        for step in range(1, 22):
            completion = llm.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=512,
            )
            response_text = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": response_text})

            code, submit = parse_response(response_text)
            if verbose:
                print(f"\nStep {step} → {'SUBMIT' if submit else 'exec'}")
                if code:
                    print(textwrap.indent(code[:250] + ("…" if len(code) > 250 else ""), "    "))

            if submit:
                result = env.step(DataCleaningAction(type="submit"))
            else:
                result = env.step(DataCleaningAction(type="exec", code=code or "pass"))

            obs = result.observation
            if verbose:
                print(f"  partial_score={obs.partial_score:.3f}  done={obs.done}")

            if obs.done:
                final_reward = obs.reward
                break

            messages.append({"role": "user", "content": build_user_message(obs, step, obs.exec_result)})

    elapsed = time.time() - start
    result_dict = {
        "task_id": task_id, "seed": seed, "model": model,
        "steps": obs.step_count, "elapsed_s": round(elapsed, 1),
        "score": final_reward,
    }
    if verbose:
        print(f"\nFINAL SCORE: {final_reward:.4f}  ({elapsed:.1f}s)")
    return result_dict

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for Data Cleaning OpenEnv")
    parser.add_argument("--model",   default="gpt-4o")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--task",    default=None,
                        choices=["ecommerce_easy","patient_records_medium","financial_audit_hard"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    from openai import OpenAI
    llm = OpenAI(api_key=api_key)

    tasks = ([args.task] if args.task
             else ["ecommerce_easy","patient_records_medium","financial_audit_hard"])

    all_results = []
    for task_id in tasks:
        r = run_episode(args.env_url, llm, args.model, task_id, args.seed,
                        verbose=not args.quiet)
        all_results.append(r)

    print(f"\n{'='*55}\n  RESULTS\n{'='*55}")
    print(f"  {'Task':<32} {'Score':>7} {'Steps':>6} {'Time':>7}")
    print(f"  {'-'*32} {'-'*7} {'-'*6} {'-'*7}")
    for r in all_results:
        print(f"  {r['task_id']:<32} {r['score']:>7.4f} {r['steps']:>6} {r['elapsed_s']:>6.1f}s")
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Average: {avg:.4f}")

    out = Path("baseline_results.json")
    out.write_text(json.dumps({"model": args.model, "seed": args.seed, "results": all_results}, indent=2))
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()