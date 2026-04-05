"""
server/dc_environment.py — DataCleaningEnvironment subclassing
openenv.core.env_server.Environment.

The simulation logic (dataset generation, sandbox, grader) lives in
environment.py and is unchanged. This file is the thin OpenEnv wrapper.
"""

from __future__ import annotations

import io as _io
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    ActionType,
    TaskID,
)
from server.environment import (
    MAX_STEPS,
    TASK_SPECS as _TASK_SPECS,
    df_to_b64,
    b64_to_df,
    generate_datasets,
    grade,
    partial_grade as _partial_grade,
    run_in_sandbox,
)


# Typed observation builder (returns DataCleaningObservation directly)

def _build_obs(
    df, task_id, gold_df, step_count, done,
    exec_result="", error=""
) -> DataCleaningObservation:
    buf = _io.StringIO()
    df.info(buf=buf)
    return DataCleaningObservation(
        df_preview    = df.head(10).to_markdown(index=False),
        df_info       = buf.getvalue(),
        df_stats      = df.describe(include="all").to_string(),
        task_spec     = _TASK_SPECS[task_id],
        exec_result   = exec_result,
        step_count    = step_count,
        partial_score = _partial_grade(df, gold_df, task_id),
        done          = done,
        error         = error,
        reward        = 0.0,
    )


# Environment
class DataCleaningEnvironment(Environment):
    """
    OpenEnv-compliant environment for tabular data cleaning.

    Three tasks of increasing difficulty:
      ecommerce_easy         — 500 rows,  type/null/format fixes
      patient_records_medium — 1 200 rows, fuzzy dedup + normalisation
      financial_audit_hard   — 5 000 rows, 10 named business rules

    Rewards are dense: partial_score updates after every exec step.
    """

    is_concurrent_safe = True

    def __init__(self) -> None:
        super().__init__()
        # Start with a blank pre-episode state.
        # All fields are Optional/defaulted so no required args are missing.
        self._state = DataCleaningState()

    # Core OpenEnv interface
    def reset(self, action: DataCleaningAction | None = None) -> DataCleaningObservation:
        """Start a new episode. Reads task_id and seed from the action."""
        task_id_str = (action.task_id if action else None) or "ecommerce_easy"
        seed        = int((action.seed if action else None) or 42)

        try:
            task_id = TaskID(task_id_str)
        except ValueError:
            task_id = TaskID.ECOMMERCE_EASY

        dirty_df, gold_df = generate_datasets(task_id, seed)

        # Pydantic models are immutable by default — always construct fresh.
        self._state = DataCleaningState(
            episode_id   = str(uuid.uuid4()),
            df_state_b64 = df_to_b64(dirty_df),
            gold_b64     = df_to_b64(gold_df),
            task_id      = task_id.value,
            seed         = seed,
            step_count   = 0,
            done         = False,
            had_crash    = False,
        )

        return _build_obs(dirty_df, task_id, gold_df, 0, False)

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        """Execute one action (exec or submit)."""
        if self._state.done:
            df   = b64_to_df(self._state.df_state_b64)
            gold = b64_to_df(self._state.gold_b64)
            return _build_obs(
                df, TaskID(self._state.task_id), gold,
                self._state.step_count, True,
                error="Episode already finished. Call reset() to start a new one.",
            )

        df      = b64_to_df(self._state.df_state_b64)
        gold_df = b64_to_df(self._state.gold_b64)
        task_id = TaskID(self._state.task_id)

        # Unpack current mutable fields from immutable state
        step_count  = self._state.step_count
        had_crash   = self._state.had_crash
        exec_result = ""
        error_msg   = ""
        final_reward = 0.0
        done        = False

        if action.type == ActionType.SUBMIT or step_count >= MAX_STEPS:
            done = True
            reward_obj   = grade(df, gold_df, task_id, step_count, had_crash)
            final_reward = reward_obj.total

        else:
            result = run_in_sandbox(action.code or "pass", df)
            exec_result = (result.stdout + result.stderr).strip()
            error_msg   = result.error

            if result.success:
                df = result.df
            else:
                had_crash = True

            step_count += 1

            if step_count >= MAX_STEPS:
                done = True
                reward_obj   = grade(df, gold_df, task_id, step_count, had_crash)
                final_reward = reward_obj.total

        # Always construct a fresh state (Pydantic immutability)
        self._state = DataCleaningState(
            episode_id   = self._state.episode_id,
            df_state_b64 = df_to_b64(df),
            gold_b64     = self._state.gold_b64,
            task_id      = self._state.task_id,
            seed         = self._state.seed,
            step_count   = step_count,
            done         = done,
            had_crash    = had_crash,
        )

        obs = _build_obs(df, task_id, gold_df, step_count, done, exec_result, error_msg)
        # reward lives on Observation in the OpenEnv spec
        return obs.model_copy(update={"reward": final_reward})

    @property
    def state(self) -> DataCleaningState:
        return self._state