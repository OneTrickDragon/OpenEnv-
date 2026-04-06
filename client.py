"""
client.py — Python client for the Data Cleaning OpenEnv environment.

Usage (sync):
    from client import DataCleaningEnv
    from models import DataCleaningAction

    with DataCleaningEnv(base_url="https://onetrickdragon-data-cleaning-openenv.hf.space").sync() as env:
        result = env.reset(task_id="ecommerce_easy", seed=42)
        print(result.observation.df_preview)
        result = env.step(DataCleaningAction(type="exec", code="df['price']=df['price'].fillna(df['price'].median())"))
        result = env.step(DataCleaningAction(type="submit"))
        print(f"Score: {result.observation.reward:.4f}")

Usage (async):
    async with DataCleaningEnv(base_url="...") as env:
        result = await env.reset()
        result = await env.step(DataCleaningAction(type="submit"))
"""

from __future__ import annotations

from typing import Optional

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server import StepResult
except ImportError:
    # Fallback for local use without package
    class EnvClient:  # type: ignore[no-redef]
        def __class_getitem__(cls, item): return cls
        def __init__(self, base_url=""): self.base_url = base_url
        def _step_payload(self, a): raise NotImplementedError
        def _parse_result(self, p): raise NotImplementedError
        def _parse_state(self, p): raise NotImplementedError
        def sync(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass
    class StepResult:  # type: ignore[no-redef]
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)


class DataCleaningEnv(EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """WebSocket client for the Data Cleaning environment."""

    def _step_payload(self, action: DataCleaningAction) -> dict:
        return {
            "type":    action.type,
            "code":    action.code,
            "task_id": action.task_id,
            "seed":    action.seed,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = DataCleaningObservation(
            df_preview    = obs_data.get("df_preview",    ""),
            df_info       = obs_data.get("df_info",       ""),
            df_stats      = obs_data.get("df_stats",      ""),
            task_spec     = obs_data.get("task_spec",     ""),
            exec_result   = obs_data.get("exec_result",   ""),
            step_count    = obs_data.get("step_count",    0),
            partial_score = obs_data.get("partial_score", 0.0),
            done          = obs_data.get("done",          False),
            reward        = obs_data.get("reward",        payload.get("reward", 0.0)),
            error         = obs_data.get("error",         ""),
        )
        return StepResult(
            observation = obs,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DataCleaningState:
        return DataCleaningState(
            episode_id   = payload.get("episode_id",   ""),
            df_state_b64 = payload.get("df_state_b64", ""),
            gold_b64     = payload.get("gold_b64",     ""),
            task_id      = payload.get("task_id",      "ecommerce_easy"),
            seed         = payload.get("seed",         42),
            step_count   = payload.get("step_count",   0),
            done         = payload.get("done",         False),
            had_crash    = payload.get("had_crash",    False),
        )

    async def reset(self, task_id: str = "ecommerce_easy", seed: int = 42) -> StepResult:
        """Start a new episode."""
        return await super().reset(DataCleaningAction(type="exec", task_id=task_id, seed=seed))

    async def exec(self, code: str) -> StepResult:
        """Execute a Python snippet."""
        return await self.step(DataCleaningAction(type="exec", code=code))

    async def submit(self) -> StepResult:
        """Submit and get the final reward."""
        return await self.step(DataCleaningAction(type="submit"))