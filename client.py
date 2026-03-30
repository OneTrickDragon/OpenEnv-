from __future__ import annotations
 
from typing import Any, Optional
 
import httpx
 
from models import (
    Action, ActionType, Observation, Reward, State, TaskID,
    ResetRequest, StepRequest,
)

class DataCleaningEnvClient:
    """
    Thin HTTP client wrapping the OpenEnv server.
 
    All public methods mirror the canonical OpenEnv interface:
        reset()  → Observation
        step()   → (Observation, Optional[Reward], bool, dict)
        state()  → State
    """
 
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self._session_id: Optional[str] = None
        self._http      = httpx.Client(timeout=timeout)

    # Core OpenEnv interface
    def reset(
        self,
        task_id: str | TaskID = TaskID.ECOMMERCE_EASY,
        seed: int = 42,
    ) -> Observation:
        """
        Start a new episode.
 
        Returns the initial Observation.  Stores the session_id internally
        so subsequent step() / state() calls are automatically routed.
        """
        if isinstance(task_id, str):
            task_id = TaskID(task_id)
 
        payload = ResetRequest(task_id=task_id, seed=seed)
        resp    = self._post("/reset", payload.model_dump())
        self._session_id = resp["session_id"]
        return Observation(**resp["observation"])
 
    def step(
        self,
        action: Action,
    ) -> tuple[Observation, Optional[Reward], bool, dict[str, Any]]:
        """
        Send one action to the environment.
 
        Returns (observation, reward, done, info).
        `reward` is only populated when done=True.
        """
        self._require_session()
        payload = StepRequest(session_id=self._session_id, action=action)
        resp    = self._post("/step", payload.model_dump())
 
        obs    = Observation(**resp["observation"])
        reward = Reward(**resp["reward"]) if resp.get("reward") else None
        done   = bool(resp["done"])
        info   = dict(resp.get("info", {}))
        return obs, reward, done, info
 
    def state(self) -> Observation:
        """
        Return the last observation — the canonical OpenEnv state() call.
 
        Use this to checkpoint the current episode state or inspect what
        the agent last saw without consuming a step.
        """
        self._require_session()
        resp = self._get(f"/observation/{self._session_id}")
        return Observation(**resp)
 
    def raw_state(self) -> State:
        """Return the internal episode state for debugging (not for training)."""
        self._require_session()
        resp = self._get(f"/state/{self._session_id}")
        return State(**resp["state"])
 
    # Convenience helpers
    def step_exec(
        self,
        code: str,
    ) -> tuple[Observation, Optional[Reward], bool, dict[str, Any]]:
        """Shorthand for step(Action(type='exec', code=code))."""
        return self.step(Action(type=ActionType.EXEC, code=code))
 
    def step_submit(self) -> tuple[Observation, Optional[Reward], bool, dict[str, Any]]:
        """Shorthand for step(Action(type='submit'))."""
        return self.step(Action(type=ActionType.SUBMIT))
 
    def available_tasks(self) -> list[dict]:
        """List all tasks with difficulty and size metadata."""
        return self._get("/tasks")["tasks"]
 
    def health(self) -> dict:
        return self._get("/health")

    # Internal helpers
    def _require_session(self) -> None:
        if not self._session_id:
            raise RuntimeError("No active session. Call reset() first.")
 
    def _post(self, path: str, body: dict) -> dict:
        r = self._http.post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()
 
    def _get(self, path: str) -> dict:
        r = self._http.get(f"{self.base_url}{path}")
        r.raise_for_status()
        return r.json()
 
    def __repr__(self) -> str:
        return (
            f"DataCleaningEnvClient(base_url={self.base_url!r}, "
            f"session_id={self._session_id!r})"
        )