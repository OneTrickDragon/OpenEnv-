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

An [OpenEnv](https://openenv.ai)-compatible reinforcement learning environment where an AI agent cleans messy tabular datasets using a sandboxed Python REPL.

## Why this environment?

Data cleaning is something every data engineer does daily. It requires diagnosis (what's wrong?), planning (what operations fix it?), and iterative refinement — exactly the skills we want agents to develop. Unlike toy problems, there's genuine ambiguity and domain knowledge required, especially in the hard task.

## Quick start

```python
from client import DataCleaningEnvClient

env = DataCleaningEnvClient(base_url="http://localhost:8000")

# Start an episode
obs = env.reset(task_id="ecommerce_easy", seed=42)
print(obs.task_spec)

# Execute cleaning steps
obs, reward, done, info = env.step_exec("""
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['price'] = df['price'].fillna(df['price'].median())
df['quantity'] = df['quantity'].clip(lower=0)
""")
print(f"Partial score: {obs.partial_score:.3f}")

# Checkpoint state without consuming a step
current_obs = env.state()

# Submit and get final reward
obs, reward, done, info = env.step_submit()
print(f"Final score: {reward.total:.4f}")
```

## Structure

```
data_cleaning_env/
├── models.py              ← Pydantic contracts: Action, Observation, State, Reward
├── client.py              ← Python client (import this in training code)
├── baseline.py            ← OpenAI-compatible inference + scoring script
├── validate.py            ← openenv validate equivalent (15 checks)
├── openenv.yaml           ← OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── tests/
│   ├── test_units.py      ← 71 unit tests (grader, sandbox, serialisation)
│   └── test_integration.py← 73 integration tests (full episode simulation)
└── server/
    ├── environment.py     ← Dataset generation, sandbox, grader
    └── app.py             ← FastAPI server
```

## Tasks

| Task | Difficulty | Rows | Cols | GPT-4o baseline |
|------|-----------|------|------|-----------------|
| `ecommerce_easy` | Easy | 500 | 8 | ~0.78 |
| `patient_records_medium` | Medium | 1,200 | 9 | ~0.55 |
| `financial_audit_hard` | Hard | 5,000 | 12 | ~0.30 |

### Task 1 — E-commerce order cleanup (Easy)
Fix type casting (`order_date` as string), impute null prices with median, clip negative quantities, normalise mixed-format revenue strings (`$12.50`, `12,50`, `12.50 USD`), strip whitespace from `customer_id`, and normalise `status` values.

### Task 2 — Patient records deduplication (Medium)
Deduplicate ~20% fuzzy duplicates (keeping the most-complete record per `patient_id`), normalise DOB to ISO-8601, phone numbers to E.164, extract ICD-10 codes from free text, lowercase emails.

### Task 3 — Financial transaction audit (Hard)
Apply 10 named business rules to 5,000 transactions — flag violations in a `violation` column, correct FX conversions, detect duplicates, drop null-required rows. Rules include temporal ordering, referential integrity, outlier detection, and currency validation.

## OpenEnv API

### Action
```python
class Action(BaseModel):
    type: "exec" | "submit"
    code: Optional[str]   # Python snippet ≤50 lines; df is in scope
```

### Observation (returned by reset, step, and state)
```python
class Observation(BaseModel):
    df_preview:    str    # first 10 rows as markdown table
    df_info:       str    # dtypes + null counts
    df_stats:      str    # df.describe() output
    task_spec:     str    # plain-English objective
    exec_result:   str    # stdout/stderr from last exec
    step_count:    int
    partial_score: float  # 0.0–1.0, updated every step (dense reward signal)
    done:          bool
    error:         str    # exception message if exec failed
```

### Reward (on episode end)
| Component | Weight | Description |
|-----------|--------|-------------|
| `column_quality` | 0.50 | Per-column dtype/null/value correctness |
| `schema_compliance` | 0.20 | Output columns match expected schema |
| `row_preservation` | 0.15 | Penalise unnecessary row drops |
| `efficiency` | 0.10 | Bonus ≤10 steps; penalty >15 steps |
| `no_crash_bonus` | 0.05 | No unhandled exceptions in episode |

### HTTP endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start episode → `{session_id, observation}` |
| `POST` | `/step` | Send action → `{observation, reward?, done, info}` |
| `GET` | `/observation/{sid}` | Last observation — canonical `state()` |
| `GET` | `/state/{sid}` | Internal episode state (debugging) |
| `GET` | `/tasks` | List tasks |
| `GET` | `/health` | Liveness probe |

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --reload --port 8000

# Run validate
python validate.py

# Run tests
python tests/test_units.py
python tests/test_integration.py

# Run baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python baseline.py
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env
```

## Sandbox constraints

| Rule | Detail |
|------|--------|
| Allowed imports | `pandas`, `numpy`, `re`, `datetime`, `difflib`, `unicodedata`, `collections`, `itertools`, `math`, `string` |
| Pre-injected | `pd`, `np`, `re`, `math`, `datetime`, `difflib`, `unicodedata`, `collections`, `itertools`, `string` |
| Blocked | `open`, file I/O, network, `subprocess`, `compile`, `breakpoint` |
| Max lines/step | 50 |
| Max steps/episode | 20 |

## Session management

Sessions expire after **1 hour of inactivity** and are hard-capped at **10,000 concurrent sessions**. Call `/reset` to start a new episode at any time.

## Baseline scores (seed=42, gpt-4o)

```
Task                           Score  Steps    Time
------------------------------ ------ ------ -------
ecommerce_easy                 0.7812      7    14.2s
patient_records_medium         0.5540     12    28.1s
financial_audit_hard           0.3021     18    51.3s

Average: 0.5458
```