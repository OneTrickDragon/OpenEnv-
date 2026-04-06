"""
server/dc_environment.py — DataCleaningEnvironment subclassing
openenv_core.env_server.Environment.
"""

from __future__ import annotations

import copy
import io as _io
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:  # type: ignore[no-redef]
        is_concurrent_safe = False
        def __init__(self): pass

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


def _build_obs(df, task_id, gold_df, step_count, done,
               exec_result="", error="") -> DataCleaningObservation:
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
        reward        = 0.0,
        error         = error,
    )


class DataCleaningEnvironment(Environment):
    """
    OpenEnv-compliant tabular data cleaning environment.

    Three tasks: ecommerce_easy / patient_records_medium / financial_audit_hard.
    Dense partial rewards after every exec step.
    """

    is_concurrent_safe = True

    def __init__(self) -> None:
        super().__init__()
        self._state = DataCleaningState()

    def reset(self, action: DataCleaningAction | None = None) -> DataCleaningObservation:
        """Start a new episode. task_id and seed come from the action."""
        task_id_str = (action.task_id if action else None) or "ecommerce_easy"
        seed        = int((action.seed if action else None) or 42)

        try:
            task_id = TaskID(task_id_str)
        except ValueError:
            task_id = TaskID.ECOMMERCE_EASY

        dirty_df, gold_df = generate_datasets(task_id, seed)

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

        df         = b64_to_df(self._state.df_state_b64)
        gold_df    = b64_to_df(self._state.gold_b64)
        task_id    = TaskID(self._state.task_id)
        step_count = self._state.step_count
        had_crash  = self._state.had_crash
        exec_result = error_msg = ""
        final_reward = 0.0
        done = False

        if action.type == ActionType.SUBMIT or step_count >= MAX_STEPS:
            done         = True
            final_reward = grade(df, gold_df, task_id, step_count, had_crash).total
        else:
            result      = run_in_sandbox(action.code or "pass", df)
            exec_result = (result.stdout + result.stderr).strip()
            error_msg   = result.error
            if result.success:
                df = result.df
            else:
                had_crash = True
            step_count += 1
            if step_count >= MAX_STEPS:
                done         = True
                final_reward = grade(df, gold_df, task_id, step_count, had_crash).total

        # Always build a fresh state (dataclasses are mutable, but constructing
        # fresh is cleaner and avoids any shared-reference bugs)
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
        obs.reward = final_reward
        return obs

    @property
    def state(self) -> DataCleaningState:
        return self._state