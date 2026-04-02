"""
client.py — Python client for the Data Cleaning OpenEnv environment.

Subclasses openenv.core.env_client.EnvClient which uses a persistent
WebSocket connection (/ws) for low-latency multi-step interaction.

Usage (async — recommended for training loops):
    import asyncio
    from client import DataCleaningEnv
    from models import DataCleaningAction

    async def main():
        async with DataCleaningEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="ecommerce_easy", seed=42)
            print(result.observation.task_spec)
            print(result.observation.df_preview)

            while not result.done:
                result = await env.step(DataCleaningAction(
                    type="exec",
                    code="df['price'] = df['price'].fillna(df['price'].median())"
                ))
                print(f"step={result.observation.step_count} "
                      f"partial_score={result.observation.partial_score:.3f}")

            print(f"Final score: {result.reward:.4f}")

    asyncio.run(main())

Usage (sync wrapper — for notebooks and scripts):
    from client import DataCleaningEnv
    from models import DataCleaningAction

    with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="ecommerce_easy", seed=42)
        result = env.step(DataCleaningAction(type="exec", code="df.dropna()"))
        result = env.step(DataCleaningAction(type="submit"))
        print(f"Final score: {result.reward:.4f}")

Connect to a live HuggingFace Space:
    with DataCleaningEnv(base_url="https://YOUR-USERNAME-data-cleaning-openenv.hf.space").sync() as env:
        result = env.reset()
"""

from __future__ import annotations

from typing import Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    TaskID,
)


class DataCleaningEnv(EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """
    WebSocket client for the Data Cleaning environment.

    Inherits reset(), step(), state(), sync(), close(), from_env(),
    and from_docker_image() from EnvClient.
    Only the payload serialisation and response parsing are environment-specific.
    """
    # Required: convert our typed action to the JSON the server expects
    def _step_payload(self, action: DataCleaningAction) -> dict:
        return {
            "type":    action.type,
            "code":    action.code,
            "task_id": action.task_id,
            "seed":    action.seed,
        }

    # Required: parse the server's JSON response into typed objects
    def _parse_result(self, payload: dict) -> StepResult[DataCleaningObservation]:
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
            error         = obs_data.get("error",         ""),
            reward        = obs_data.get("reward",        payload.get("reward", 0.0)),
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

    async def reset(
        self,
        task_id: str = "ecommerce_easy",
        seed:    int = 42,
    ) -> StepResult[DataCleaningObservation]:
        """Start a new episode. Passes task_id and seed via a reset action."""
        action = DataCleaningAction(type="exec", task_id=task_id, seed=seed)
        return await super().reset(action)

    async def exec(self, code: str) -> StepResult[DataCleaningObservation]:
        """Shorthand: execute a Python snippet."""
        return await self.step(DataCleaningAction(type="exec", code=code))

    async def submit(self) -> StepResult[DataCleaningObservation]:
        """Shorthand: submit the current DataFrame and get the final reward."""
        return await self.step(DataCleaningAction(type="submit"))