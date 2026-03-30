"""
server/app.py — FastAPI server exposing the OpenEnv HTTP interface.
 
Endpoints:
  POST /reset          → start a new episode, return session_id + initial observation
  POST /step           → send an action, receive observation + optional reward
  GET  /state/{sid}    → inspect raw episode state (debugging / evaluation)
  GET  /health         → liveness probe
  GET  /tasks          → list available tasks
"""

from __future__ import annotations
 
import uuid
from typing import Any
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from models import (
    Action, ActionType, Observation, Reward, State, TaskID,
    ResetRequest, ResetResponse, StepRequest, StepResponse, StateResponse,
)
from server.environment import (
    MAX_STEPS, build_observation, df_to_b64, b64_to_df,
    generate_datasets, grade, run_in_sandbox,
)

app = FastAPI(
    title       = "Data Cleaning OpenEnv",
    description = "An OpenEnv-compatible environment for training agents to clean tabular data.",
    version     = "0.1.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

import time as _time
 
SESSION_TTL_SECONDS = 3600        
SESSION_MAX         = 10_000      
 
sessions: dict[str, dict[str, Any]] = {}
 
 
def _evict_expired() -> None:
    """Remove sessions idle longer than SESSION_TTL_SECONDS."""
    now     = _time.monotonic()
    expired = [sid for sid, s in sessions.items()
               if now - s["last_access"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del sessions[sid]
    if len(sessions) > SESSION_MAX:
        oldest = sorted(sessions, key=lambda sid: sessions[sid]["last_access"])
        for sid in oldest[:len(sessions) - SESSION_MAX]:
            del sessions[sid]
 
 
def _get_session(session_id: str) -> dict[str, Any]:
    _evict_expired()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    sess = sessions[session_id]
    sess["last_access"] = _time.monotonic()
    return sess

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions_active": len(sessions)}
 
 
@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": [
            {"id": "ecommerce_easy",         "difficulty": "easy",   "rows": 500,  "cols": 8},
            {"id": "patient_records_medium", "difficulty": "medium", "rows": 1200, "cols": 9},
            {"id": "financial_audit_hard",   "difficulty": "hard",   "rows": 5000, "cols": 12},
        ]
    }

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    """
    Start a new episode.  Returns a session_id and the initial observation.
    The agent should store the session_id and pass it with every /step call.
    """
    dirty_df, gold_df = generate_datasets(req.task_id, req.seed)
 
    state = State(
        df_state_b64 = df_to_b64(dirty_df),
        task_id        = req.task_id,
        seed           = req.seed,
        step_count     = 0,
        done           = False,
        had_crash      = False,
    )
 
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "state":       state,
        "gold_b64":    df_to_b64(gold_df),
        "last_obs":    obs,
        "last_access": _time.monotonic(),
    }
 
    obs = build_observation(
        df         = dirty_df,
        task_id    = req.task_id,
        gold_df    = gold_df,
        step_count = 0,
        done       = False,
    )
 
    return ResetResponse(session_id=session_id, observation=obs)

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """
    Execute one action in the environment.
 
    - action.type = "exec":   run code, return updated observation + partial_score.
    - action.type = "submit": finalise episode, return full Reward.
 
    The episode also ends automatically after MAX_STEPS exec steps.
    """
    sess     = _get_session(req.session_id)
    state: State = sess["state"]
 
    if state.done:
        raise HTTPException(status_code=400, detail="Episode already finished. Call /reset to start a new one.")
 
    df      = b64_to_df(state.df_state_b64)
    gold_df = b64_to_df(sess["gold_b64"])
    action  = req.action
 
    exec_result = ""
    error_msg   = ""
    reward: Reward | None = None

    if action.type == ActionType.EXEC:
        if not action.code:
            raise HTTPException(status_code=422, detail="action.code is required for type='exec'")
 
        result = run_in_sandbox(action.code, df)
        exec_result = (result.stdout + result.stderr).strip()
        error_msg   = result.error
 
        if result.success:
            df = result.df
        else:
            state.had_crash = True
 
        state.step_count += 1
        state.df_state_b64 = df_to_b64(df)

        if state.step_count >= MAX_STEPS:
            state.done = True
            reward = grade(df, gold_df, state.task_id, state.step_count, state.had_crash)
 
    elif action.type == ActionType.SUBMIT:
        state.done = True
        reward = grade(df, gold_df, state.task_id, state.step_count, state.had_crash)

    obs = build_observation(
        df         = df,
        task_id    = state.task_id,
        gold_df    = gold_df,
        step_count = state.step_count,
        done       = state.done,
        exec_result= exec_result,
        error      = error_msg,
    )
 
    # Persist updated state and last observation
    sess["state"]    = state
    sess["last_obs"] = obs
 
    info: dict[str, Any] = {"step_count": state.step_count}
    if reward:
        info["reward_breakdown"] = reward.breakdown
 
    return StepResponse(
        observation = obs,
        reward      = reward,
        done        = state.done,
        info        = info,
    )

@app.get("/state/{session_id}", response_model=StateResponse)
def get_state(session_id: str) -> StateResponse:
    """
    Return the current episode state.
 
    The OpenEnv spec defines state() as returning a snapshot that can be used
    to resume or inspect the episode. Returns the last Observation alongside
    the internal State so training code can checkpoint without re-running steps.
    """
    sess = _get_session(session_id)
    return StateResponse(session_id=session_id, state=sess["state"])

@app.get("/observation/{session_id}", response_model=Observation)
def get_observation(session_id: str) -> Observation:
    """
    Return the last observation — the canonical state() in OpenEnv terms.
 
    Training loops should use this endpoint (via client.state()) to retrieve
    the most recent observation for checkpointing or inspection.
    """
    sess = _get_session(session_id)
    return sess["last_obs"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)