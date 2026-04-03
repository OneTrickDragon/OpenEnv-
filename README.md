---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - data-engineering
  - tabular
space_sdk_type: openenv
---

# Data Cleaning OpenEnv

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment
where an AI agent cleans messy tabular datasets using a sandboxed Python REPL.

## Install

```bash
# Install from this HF Space directly
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/data-cleaning-openenv

# Or install openenv-core and clone
pip install "openenv-core[core]>=0.2.1"
```

## Quick start (sync)

```python
from client import DataCleaningEnv
from models import DataCleaningAction

with DataCleaningEnv(base_url="https://YOUR_USERNAME-data-cleaning-openenv.hf.space").sync() as env:
    result = env.reset(task_id="ecommerce_easy", seed=42)
    print(result.observation.task_spec)

    result = env.step(DataCleaningAction(
        type="exec",
        code="df['price'] = df['price'].fillna(df['price'].median())"
    ))
    print(f"partial_score={result.observation.partial_score:.3f}")

    result = env.step(DataCleaningAction(type="submit"))
    print(f"Final score: {result.reward:.4f}")
```

## Quick start (async)

```python
import asyncio
from client import DataCleaningEnv
from models import DataCleaningAction

async def run():
    async with DataCleaningEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="ecommerce_easy", seed=42)
        result = await env.exec("df['price'] = df['price'].fillna(df['price'].median())")
        result = await env.submit()
        print(f"Score: {result.reward:.4f}")

asyncio.run(run())
```

## Run locally

```bash
# Clone and install
git clone https://huggingface.co/spaces/YOUR_USERNAME/data-cleaning-openenv
cd data-cleaning-openenv
pip install -e .

# Start server
uvicorn server.app:app --port 8000

# Or via Docker
docker build -t data-cleaning-openenv .
docker run -p 8000:7860 data-cleaning-openenv
```

## Tasks

| Task | Difficulty | Rows | Cols | GPT-4o baseline |
|------|-----------|------|------|-----------------|
| `ecommerce_easy` | Easy | 500 | 8 | ~0.78 |
| `patient_records_medium` | Medium | 1 200 | 9 | ~0.55 |
| `financial_audit_hard` | Hard | 5 000 | 12 | ~0.30 |

## Action space

```python
@dataclass
class DataCleaningAction(Action):
    type:    str  = "exec"           # "exec" or "submit"
    code:    str  = None             # Python snippet ≤50 lines; df in scope
    task_id: str  = "ecommerce_easy" # used only in reset()
    seed:    int  = 42               # used only in reset()
```

## Observation space

```python
@dataclass
class DataCleaningObservation(Observation):
    df_preview:    str    # first 10 rows as markdown table
    df_info:       str    # dtypes + null counts
    df_stats:      str    # df.describe() output
    task_spec:     str    # plain-English objective
    exec_result:   str    # stdout/stderr from last exec
    step_count:    int    # steps used so far
    partial_score: float  # 0.0–1.0, updated every step (dense reward)
    done:          bool
    error:         str    # exception message if exec failed
    reward:        float  # final reward when done=True
```

## Reward

| Component | Weight | Description |
|-----------|--------|-------------|
| `column_quality` | 0.50 | Per-column dtype/null/value correctness |
| `schema_compliance` | 0.20 | Output columns match expected schema |
| `row_preservation` | 0.15 | Penalise unnecessary row drops |
| `efficiency` | 0.10 | Bonus ≤10 steps; penalty >15 steps |
| `no_crash_bonus` | 0.05 | No unhandled exceptions in episode |

## Sandbox constraints

| Rule | Detail |
|------|--------|
| Allowed imports | `pandas`, `numpy`, `re`, `datetime`, `difflib`, `unicodedata`, `collections`, `itertools`, `math`, `string` |
| Blocked | `open`, file I/O, network, `subprocess` |
| Max lines/step | 50 |
| Max steps/episode | 20 |

## Deploy with OpenEnv CLI

```bash
pip install "openenv-core[core]"
openenv push --repo-id YOUR_USERNAME/data-cleaning-openenv
```"# trigger rebuild" 
