"""SandboxScoringRunner — wrap any AgentRunner with a per-task sandbox.

Flow for one sandbox task:

    1. resolve the sandbox spec from ``TaskInput.sandbox`` and start the
       environment (temp dir / container);
    2. optional ``setup_fn(task, sandbox)`` prepares the environment (e.g. a
       local fixture does ``git init``; a docker SWE-bench image needs nothing);
    3. inject the *live* sandbox into ``initial_state["__sandbox__"]`` so the
       agent's (unchanged) ``binding.tool_executor`` reaches it, then run the
       inner agent — it edits code inside the sandbox via bash / editor tools;
    4. extract the model's diff (``git diff`` in the sandbox's working dir);
    5. while the sandbox is still alive, run the dataset's ``score_fn`` (apply
       patch + run tests) and stash ``resolved`` / report into ``TaskTrace.raw``.

This is the seam that lets every existing agent work on code-execution
datasets without a single change: the agent only ever calls
``binding.tool_executor``; the sandbox arrives through ``state["__sandbox__"]``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ageneval.task.core.agent import AgentRunner
from ageneval.task.core.async_utils import run_sync_in_daemon_thread
from ageneval.task.core.dataset import TaskInput
from ageneval.task.core.grading import GradeReport, GraderSpec
from ageneval.task.core.result import TaskTrace

if TYPE_CHECKING:
    from ageneval.task.sandbox import SandboxEnvironment

logger = logging.getLogger(__name__)

# (task, sandbox) -> None; (task, sandbox, model_patch) -> report dict
SetupFn = Callable[[TaskInput, "SandboxEnvironment"], None]
ScoreFn = Callable[[TaskInput, "SandboxEnvironment", str], Mapping[str, Any]]
LifecycleHook = Callable[[str], Awaitable[None]]


@dataclass
class SandboxScoringRunner(AgentRunner):
    """Run an inner agent inside a per-task sandbox and score the result."""

    inner: AgentRunner
    score_fn: ScoreFn | None = None
    grader: GraderSpec | None = None
    setup_fn: SetupFn | None = None
    patch_cmd: Sequence[str] = ("git", "diff")
    lifecycle_hook: LifecycleHook | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.score_fn is None) == (self.grader is None):
            raise ValueError("provide exactly one of score_fn or grader")
        if self.grader is not None and self.grader.mode != "inline":
            raise ValueError("sandbox grader must use mode='inline'")
        self.name = f"sandbox::{getattr(self.inner, 'name', 'agent')}"

    async def run(self, task: TaskInput) -> TaskTrace:
        from ageneval.task.sandbox import SandboxSpec, sandbox_session

        start = time.perf_counter()
        spec = SandboxSpec.from_obj(task.sandbox)
        if spec is None:
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error="sandbox dataset task is missing a 'sandbox' spec",
            )

        environment_started = False
        try:
            if self.lifecycle_hook is not None:
                await self.lifecycle_hook("ENVIRONMENT_START")
            environment_started = True
            with sandbox_session(spec) as sb:
                if self.setup_fn is not None:
                    self.setup_fn(task, sb)
                inner_task = replace(
                    task,
                    initial_state={**dict(task.initial_state), "__sandbox__": sb},
                )
                agent_timeout = task.metadata.get("agent_timeout_sec")
                try:
                    if self.lifecycle_hook is not None:
                        await self.lifecycle_hook("AGENT_START")
                    try:
                        if agent_timeout is None:
                            trace = await self.inner.run(inner_task)
                        else:
                            timeout = float(agent_timeout)
                            if timeout <= 0:
                                raise ValueError("agent_timeout_sec must be positive")
                            trace = await asyncio.wait_for(
                                self.inner.run(inner_task), timeout=timeout
                            )
                    finally:
                        if self.lifecycle_hook is not None:
                            await self.lifecycle_hook("AGENT_END")
                except asyncio.TimeoutError:
                    logger.warning("agent timed out on %s after %ss", task.task_id, agent_timeout)
                    trace = TaskTrace(
                        task_id=task.task_id,
                        agent_name=getattr(self.inner, "name", "agent"),
                        status="error",
                        turns=0,
                        elapsed_seconds=time.perf_counter() - start,
                        error=f"agent timed out after {agent_timeout}s",
                        raw={"agent_timed_out": True},
                    )
                model_patch = sb.exec(list(self.patch_cmd)).stdout
                try:
                    if self.lifecycle_hook is not None:
                        await self.lifecycle_hook("VERIFICATION_START")
                    # Verifiers execute synchronous sandbox commands and may run
                    # for many minutes.  Running one directly on the asyncio event
                    # loop freezes every other task: agent deadlines cannot fire
                    # and completed slots cannot schedule their next sample.
                    # Keep the sandbox session alive here, but move the blocking
                    # scorer to a daemon worker. A cancelled default-executor
                    # worker would otherwise delay asyncio/interpreter shutdown.
                    scorer = (
                        self.grader.resolve()
                        if self.grader is not None
                        else self.score_fn
                    )
                    assert scorer is not None
                    grade_value = await run_sync_in_daemon_thread(
                        scorer,
                        task,
                        sb,
                        model_patch,
                        thread_name=f"a2e-scorer-{task.task_id}",
                    )
                    if self.grader is not None:
                        normalized = self.grader.summarize_inline(grade_value)
                        report = (
                            dict(grade_value)
                            if isinstance(grade_value, Mapping)
                            else {}
                        )
                        report["grade_report"] = normalized.as_dict()
                    elif isinstance(grade_value, GradeReport):
                        report = {"grade_report": grade_value.as_dict()}
                    else:
                        report = dict(grade_value)
                except Exception as exc:  # scoring must not crash the run
                    logger.exception("scorer failed on %s", task.task_id)
                    report = {"resolved": False, "score_error": str(exc)[:500]}
                finally:
                    if self.lifecycle_hook is not None:
                        await self.lifecycle_hook("VERIFICATION_END")
        except Exception as exc:  # sandbox provisioning failure
            logger.exception("sandbox failed on %s", task.task_id)
            return TaskTrace(
                task_id=task.task_id,
                agent_name=self.name,
                status="error",
                turns=0,
                elapsed_seconds=time.perf_counter() - start,
                error=f"sandbox error: {exc}"[:1000],
            )
        finally:
            if environment_started and self.lifecycle_hook is not None:
                await self.lifecycle_hook("ENVIRONMENT_END")

        raw = {**dict(trace.raw), "model_patch": model_patch, **report}
        return replace(
            trace,
            agent_name=self.name,
            raw=raw,
            elapsed_seconds=trace.elapsed_seconds or (time.perf_counter() - start),
        )
