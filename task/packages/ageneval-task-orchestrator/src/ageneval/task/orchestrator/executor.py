"""Cell worker process and per-trial lifecycle execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from ageneval.model.gateway import ResolvedModel

from .schema import (
    GraderConfig,
    GradeResult,
    LifecycleEvent,
    LifecycleRecord,
    RetryPolicy,
    TrialResult,
)

logger = logging.getLogger(__name__)


LifecycleHook = Callable[[LifecycleEvent, int | None], Awaitable[None]]


def _trace_output(trace: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "final_answer": trace.final_answer or "",
        "tool_calls": [call.name for call in trace.tool_calls],
        "tool_call_records": [
            {
                "name": call.name,
                "arguments": dict(call.arguments),
                "result": call.result,
                "error": call.error,
            }
            for call in trace.tool_calls
        ],
        "status": trace.status,
        "turns": trace.turns,
        "elapsed_seconds": trace.elapsed_seconds,
        "error": trace.error,
    }
    output.update(dict(trace.raw))
    if "model_patch" in output:
        output["model_patch"] = str(output["model_patch"] or "")[:4000]
    return json.loads(json.dumps(output, ensure_ascii=False, default=str))


def _normalize_grade(
    grader: GraderConfig,
    value: Any,
    *,
    started_at: datetime,
    ended_at: datetime,
    trace_id: str | None,
) -> GradeResult:
    from ageneval.task.core.grading import GradeReport

    if isinstance(value, GradeReport):
        metadata = {
            **dict(value.metadata),
            "metrics": dict(value.metrics),
            "official": value.official,
            "source": value.source,
            "version": value.version,
            "passed": value.passed,
        }
        return GradeResult(
            name=grader.id,
            mode=grader.mode,
            required=grader.required,
            annotator_kind=(
                "LLM" if grader.id in {"gdp_grader", "llm_judge"} else "CODE"
            ),
            score=value.score,
            label=value.label
            or (
                "pass"
                if value.passed is True
                else "fail"
                if value.passed is False
                else None
            ),
            explanation=value.explanation,
            metadata=metadata,
            error=value.error,
            start_time=started_at,
            end_time=ended_at,
            trace_id=trace_id,
        )
    if isinstance(value, bool):
        score = float(value)
        return GradeResult(
            name=grader.id,
            mode=grader.mode,
            required=grader.required,
            annotator_kind="LLM" if grader.id == "llm_judge" else "CODE",
            score=score,
            label="pass" if value else "fail",
            start_time=started_at,
            end_time=ended_at,
            trace_id=trace_id,
        )
    if isinstance(value, (int, float)):
        return GradeResult(
            name=grader.id,
            mode=grader.mode,
            required=grader.required,
            annotator_kind="LLM" if grader.id == "llm_judge" else "CODE",
            score=float(value),
            start_time=started_at,
            end_time=ended_at,
            trace_id=trace_id,
        )
    if isinstance(value, dict):
        result = value.get("result") if isinstance(value.get("result"), dict) else value
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        metadata = {
            **metadata,
            **{
                key: item
                for key, item in value.items()
                if key not in {"score", "label", "explanation", "metadata", "result", "error"}
            },
        }
        return GradeResult(
            name=grader.id,
            mode=grader.mode,
            required=grader.required,
            annotator_kind="LLM" if grader.id == "llm_judge" else "CODE",
            score=result.get("score"),
            label=result.get("label"),
            explanation=result.get("explanation"),
            metadata=metadata,
            error=value.get("error"),
            start_time=started_at,
            end_time=ended_at,
            trace_id=trace_id,
        )
    return GradeResult(
        name=grader.id,
        mode=grader.mode,
        required=grader.required,
        annotator_kind="LLM" if grader.id == "llm_judge" else "CODE",
        label=str(value),
        start_time=started_at,
        end_time=ended_at,
        trace_id=trace_id,
    )


async def _run_grader(
    grader: GraderConfig,
    *,
    benchmark_id: str,
    task: Any,
    output: dict[str, Any],
    trace_id: str | None,
    resolved_model: Any | None = None,
) -> GradeResult:
    from ageneval.task.core.grading import normalize_grade, run_grader
    from ageneval.task.runners import grader_for_dataset

    started_at = datetime.now(timezone.utc)
    try:
        spec = grader_for_dataset(benchmark_id)
        if grader.id not in {spec.id, *spec.aliases}:
            raise ValueError(
                f"benchmark {benchmark_id!r} owns grader {spec.id!r}, "
                f"not {grader.id!r}"
            )
        if spec.mode == "inline":
            inline_error = (
                output.get("score_error")
                or output.get("tb_reward_read_error")
                or output.get("tb_ctrf_error")
            )
            if inline_error:
                raise RuntimeError(str(inline_error))
            embedded = output.get("grade_report")
            if not isinstance(embedded, dict):
                raise ValueError(
                    f"inline grader {spec.id!r} did not produce grade_report"
                )
            value = normalize_grade(embedded, spec)
        else:
            runtime = None
            if spec.factory is not None:
                if resolved_model is None:
                    raise ValueError(f"{spec.id} requires a resolved model runtime")
                from a2e.evals.llm import LLM

                provider = (
                    "anthropic"
                    if resolved_model.profile.upstream_protocol.value == "anthropic_messages"
                    else "openai"
                )
                llm_kwargs: dict[str, Any] = {
                    "provider": provider,
                    "model": grader.model or resolved_model.profile.model,
                    "api_key": resolved_model.api_key.get_secret_value(),
                }
                if resolved_model.base_url:
                    llm_kwargs["base_url"] = resolved_model.base_url
                runtime = LLM(**llm_kwargs)
            value = await run_grader(
                spec,
                output=output,
                expected={
                    "expected_outputs": list(task.expected_outputs),
                    "expected_actions": list(task.expected_actions),
                },
                input={
                    "instruction": task.instruction,
                    "initial_state": dict(task.initial_state),
                },
                metadata=dict(task.metadata),
                example=task,
                runtime=runtime,
            )
        ended_at = datetime.now(timezone.utc)
        return _normalize_grade(
            grader,
            value,
            started_at=started_at,
            ended_at=ended_at,
            trace_id=trace_id,
        )
    except Exception as exc:
        return GradeResult(
            name=grader.id,
            mode=grader.mode,
            required=grader.required,
            annotator_kind="LLM" if grader.id == "llm_judge" else "CODE",
            error=str(exc)[:1000],
            start_time=started_at,
            end_time=datetime.now(timezone.utc),
            trace_id=trace_id,
        )


def _error_retryable(error_type: str, policy: RetryPolicy) -> bool:
    if error_type in policy.exclude:
        return False
    if policy.include:
        return error_type in policy.include
    return error_type in {
        "ConnectionError",
        "RateLimitError",
        "ServerError",
        "SandboxProvisioningError",
    }


def _classify_trace_error(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "rate limit" in lowered or "429" in lowered:
        return "RateLimitError"
    if "connection" in lowered or "temporarily unavailable" in lowered:
        return "ConnectionError"
    if "503" in lowered or "502" in lowered or "server error" in lowered:
        return "ServerError"
    if "sandbox error" in lowered or ("container" in lowered and "start" in lowered):
        return "SandboxProvisioningError"
    return "AgentError"


async def _execute_isolated_attempt(
    *,
    payload: dict[str, Any],
    trial: dict[str, Any],
    task_payload: dict[str, Any],
    resolved_model: ResolvedModel,
    lifecycle: LifecycleHook,
    attempt: int,
) -> TrialResult:
    started_at = datetime.now(timezone.utc)
    process = None
    environment_started = False
    agent_started = False
    verification_started = False
    try:
        # The external interpreter cannot report its inner phase boundaries.
        # Hold conservative lifecycle permits for its full process lifetime;
        # upper bounds remain exact even though utilization is less granular.
        if task_payload.get("sandbox") is not None:
            await lifecycle(LifecycleEvent.ENVIRONMENT_START, attempt)
            environment_started = True
        await lifecycle(LifecycleEvent.AGENT_START, attempt)
        agent_started = True
        if payload["benchmark"].get("graders"):
            await lifecycle(LifecycleEvent.VERIFICATION_START, attempt)
            verification_started = True
        child_payload = {
            "cell": payload["cell"],
            "trial": trial,
            "task": task_payload,
            "benchmark": payload["benchmark"],
            "profile": resolved_model.profile.public_dict(),
            "base_url": resolved_model.base_url,
            "project_name": payload["project_name"],
            "otel_endpoint": payload.get("otel_endpoint"),
            "attempt": attempt,
            "timeout_seconds": payload.get("timeout_seconds"),
        }
        env = dict(os.environ)
        env["A2E_ISOLATED_MODEL_API_KEY"] = resolved_model.api_key.get_secret_value()
        python_path = payload["isolated_pythonpath"]
        if env.get("PYTHONPATH"):
            python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = python_path
        process = await asyncio.create_subprocess_exec(
            payload["isolated_python"],
            payload["isolated_script"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        encoded = json.dumps(child_payload, ensure_ascii=False).encode()
        timeout = payload.get("timeout_seconds")
        communicate = process.communicate(encoded)
        if timeout is None:
            stdout, _stderr = await communicate
        else:
            stdout, _stderr = await asyncio.wait_for(communicate, timeout=timeout + 30)
        if process.returncode != 0:
            raise RuntimeError(f"isolated worker exited with code {process.returncode}")
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("isolated worker returned no result")
        return TrialResult.model_validate_json(lines[-1])
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        raise
    except Exception as exc:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        return TrialResult(
            trial_id=trial["trial_id"],
            cell_id=payload["cell"]["cell_id"],
            task_id=trial["task_id"],
            repetition=trial["repetition"],
            attempt=attempt,
            status="failed",
            error=str(exc)[:2000],
            error_type=type(exc).__name__,
            retryable=_error_retryable(
                type(exc).__name__, RetryPolicy.model_validate(payload["cell"]["retry"])
            ),
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
        )
    finally:
        if verification_started:
            await lifecycle(LifecycleEvent.VERIFICATION_END, attempt)
        if agent_started:
            await lifecycle(LifecycleEvent.AGENT_END, attempt)
        if environment_started:
            await lifecycle(LifecycleEvent.ENVIRONMENT_END, attempt)


async def _execute_attempt(
    *,
    cell: dict[str, Any],
    trial: dict[str, Any],
    task_payload: dict[str, Any],
    benchmark: dict[str, Any],
    resolved_model: Any,
    provider: Any,
    lifecycle: LifecycleHook,
    attempt: int,
    timeout_seconds: float | None,
    agent_kwargs_override: dict[str, Any] | None = None,
) -> TrialResult:
    from ageneval.task.core import SandboxScoringRunner, TaskInput
    from ageneval.task.runners import (
        AGENTS,
        DATASETS,
        grader_for_dataset,
        wrap_agent_for_dataset,
    )
    from opentelemetry.trace import Status, StatusCode

    started_at = datetime.now(timezone.utc)

    async def hook(event: str) -> None:
        await lifecycle(LifecycleEvent(event), attempt)

    task = TaskInput(
        task_id=task_payload["task_id"],
        instruction=task_payload["instruction"],
        initial_state=task_payload.get("initial_state") or {},
        expected_actions=task_payload.get("expected_actions") or [],
        expected_outputs=task_payload.get("expected_outputs") or [],
        metadata=task_payload.get("metadata") or {},
        sandbox=task_payload.get("sandbox"),
    )
    tracer = provider.get_tracer("ageneval.task.orchestrator")
    trace_id: str | None = None
    try:
        ds_entry = DATASETS[cell["benchmark"]]
        bind_kwargs: dict[str, Any] = {}
        if benchmark.get("domain"):
            bind_kwargs["domain"] = benchmark["domain"]
        binding = ds_entry["bind"](**bind_kwargs)
        agent_kwargs = (
            dict(agent_kwargs_override)
            if agent_kwargs_override is not None
            else resolved_model.agent_kwargs()
        )
        for key, value in (ds_entry.get("agent_overrides") or {}).items():
            agent_kwargs.setdefault(key, value)
        agent = AGENTS[cell["harness"]]["build"](binding=binding, **agent_kwargs)
        runner: Any = wrap_agent_for_dataset(cell["benchmark"], agent)
        sandbox = ds_entry.get("kind") == "sandbox"
        grader_items = list(benchmark.get("graders") or [])
        grader_spec = (
            grader_for_dataset(cell["benchmark"])
            if sandbox or grader_items
            else None
        )
        if sandbox:
            assert grader_spec is not None
            runner = SandboxScoringRunner(
                inner=agent,
                grader=grader_spec,
                setup_fn=ds_entry.get("setup"),
                lifecycle_hook=hook,
            )

        async def run_body() -> tuple[Any, list[GradeResult]]:
            nonlocal trace_id
            span_name = f"campaign.trial.{trial['trial_id']}"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("a2e.campaign_id", cell["campaign_id"])
                span.set_attribute("a2e.cell_id", cell["cell_id"])
                span.set_attribute("a2e.task_id", task.task_id)
                trace_id = f"{span.get_span_context().trace_id:032x}"
                if sandbox:
                    trace = await runner.run(task)
                else:
                    await hook("AGENT_START")
                    try:
                        trace = await runner.run(task)
                    finally:
                        await hook("AGENT_END")
                output = _trace_output(trace)
                grades: list[GradeResult] = []
                graders = [GraderConfig.model_validate(item) for item in benchmark["graders"]]
                if graders:
                    await hook("VERIFICATION_START")
                    try:
                        for grader in graders:
                            grades.append(
                                await _run_grader(
                                    grader,
                                    benchmark_id=cell["benchmark"],
                                    task=task,
                                    output=output,
                                    trace_id=trace_id,
                                    resolved_model=resolved_model,
                                )
                            )
                    finally:
                        await hook("VERIFICATION_END")
                if trace.error:
                    span.set_status(Status(StatusCode.ERROR, trace.error))
                return trace, grades

        if timeout_seconds is None:
            trace, grades = await run_body()
        else:
            trace, grades = await asyncio.wait_for(run_body(), timeout=timeout_seconds)
        output = _trace_output(trace)
        required_grade_error = next(
            (grade.error for grade in grades if grade.required and grade.error), None
        )
        error = trace.error or required_grade_error
        error_type = _classify_trace_error(trace.error) or (
            "GraderError" if required_grade_error else None
        )
        return TrialResult(
            trial_id=trial["trial_id"],
            cell_id=cell["cell_id"],
            task_id=trial["task_id"],
            repetition=trial["repetition"],
            attempt=attempt,
            status="failed" if error else "local_complete",
            output=output,
            grades=grades,
            trace_id=trace_id,
            error=error,
            error_type=error_type,
            retryable=_error_retryable(error_type or "", RetryPolicy.model_validate(cell["retry"])),
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_type = type(exc).__name__
        policy = RetryPolicy.model_validate(cell["retry"])
        logger.exception("trial %s failed", trial["trial_id"])
        return TrialResult(
            trial_id=trial["trial_id"],
            cell_id=cell["cell_id"],
            task_id=trial["task_id"],
            repetition=trial["repetition"],
            attempt=attempt,
            status="failed",
            error=str(exc)[:2000],
            error_type=error_type,
            retryable=_error_retryable(error_type, policy),
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
        )


async def _run_cell_async(payload: dict[str, Any], command_queue: Any, event_queue: Any) -> None:
    from ageneval.task.core import setup_instrumentation
    from ageneval.task.runners import framework_for_agent

    runtime_model = ResolvedModel.model_validate(payload["resolved_model"])
    provider = None
    reply_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
    trial_tasks: dict[str, asyncio.Task[None]] = {}

    async def send(message: dict[str, Any]) -> None:
        # The bounded IPC queue is the intentional backpressure boundary.
        # A direct put avoids leaving default-executor queue readers/writers
        # behind during asyncio.run() shutdown in a spawned Worker process.
        event_queue.put(message)
        await asyncio.sleep(0)

    async def run_trial(trial: dict[str, Any]) -> None:
        records: list[LifecycleRecord] = []
        attempt = int(trial.get("attempt", 1))
        result: TrialResult | None = None

        async def lifecycle(event: LifecycleEvent, event_attempt: int | None) -> None:
            timestamp = datetime.now(timezone.utc)
            records.append(LifecycleRecord(event=event, timestamp=timestamp, attempt=event_attempt))
            request_id = uuid.uuid4().hex
            future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            reply_futures[request_id] = future
            await send(
                {
                    "type": "lifecycle",
                    "request_id": request_id,
                    "cell_id": payload["cell"]["cell_id"],
                    "trial_id": trial["trial_id"],
                    "event": event.value,
                    "attempt": event_attempt,
                    "timestamp": timestamp.isoformat(),
                }
            )
            try:
                reply = await future
            except asyncio.CancelledError:
                reply_futures.pop(request_id, None)
                # This is advisory; the Controller also releases every lease
                # by trial/cell when a worker disappears.
                event_queue.put(
                    {
                        "type": "lifecycle_abandon",
                        "request_id": request_id,
                        "cell_id": payload["cell"]["cell_id"],
                        "trial_id": trial["trial_id"],
                    }
                )
                raise
            finally:
                reply_futures.pop(request_id, None)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or "lifecycle request failed")

        policy = RetryPolicy.model_validate(payload["cell"]["retry"])
        started_at = datetime.now(timezone.utc)
        try:
            # Like Harbor's TrialQueue, START owns the global Trial permit for
            # the complete retry loop, including exponential backoff.
            await lifecycle(LifecycleEvent.START, None)
            while True:
                if payload.get("isolated"):
                    result = await _execute_isolated_attempt(
                        payload=payload,
                        trial=trial,
                        task_payload=payload["tasks_by_id"][trial["task_id"]],
                        resolved_model=runtime_model,
                        lifecycle=lifecycle,
                        attempt=attempt,
                    )
                else:
                    result = await _execute_attempt(
                        cell=payload["cell"],
                        trial=trial,
                        task_payload=payload["tasks_by_id"][trial["task_id"]],
                        benchmark=payload["benchmark"],
                        resolved_model=runtime_model,
                        provider=provider,
                        lifecycle=lifecycle,
                        attempt=attempt,
                        timeout_seconds=payload.get("timeout_seconds"),
                    )
                final = not (
                    result.status == "failed" and result.retryable and attempt <= policy.max_retries
                )
                if final:
                    break
                await send(
                    {
                        "type": "trial_result",
                        "result": result.model_copy(update={"lifecycle": list(records)}).model_dump(
                            mode="json"
                        ),
                        "final": False,
                    }
                )
                wait_seconds = min(
                    policy.max_wait_seconds,
                    policy.min_wait_seconds * (policy.multiplier ** max(0, attempt - 1)),
                )
                await asyncio.sleep(wait_seconds)
                attempt += 1
        except asyncio.CancelledError:
            try:
                await lifecycle(LifecycleEvent.CANCEL, attempt)
            except Exception:
                logger.exception("failed to emit CANCEL for %s", trial["trial_id"])
            result = TrialResult(
                trial_id=trial["trial_id"],
                cell_id=payload["cell"]["cell_id"],
                task_id=trial["task_id"],
                repetition=trial["repetition"],
                attempt=attempt,
                status="cancelled",
                error="trial cancelled",
                error_type="CancelledError",
                started_at=started_at,
                ended_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.exception("trial lifecycle %s failed", trial["trial_id"])
            result = TrialResult(
                trial_id=trial["trial_id"],
                cell_id=payload["cell"]["cell_id"],
                task_id=trial["task_id"],
                repetition=trial["repetition"],
                attempt=attempt,
                status="failed",
                error=str(exc)[:2000],
                error_type=type(exc).__name__,
                retryable=False,
                started_at=started_at,
                ended_at=datetime.now(timezone.utc),
            )
        finally:
            try:
                await lifecycle(LifecycleEvent.END, attempt)
            except Exception:
                logger.exception("failed to emit END for %s", trial["trial_id"])
        if result is None:  # pragma: no cover - defensive state invariant
            raise RuntimeError(f"trial {trial['trial_id']} produced no result")
        result = result.model_copy(
            update={"lifecycle": list(records), "ended_at": datetime.now(timezone.utc)}
        )
        await send(
            {
                "type": "trial_result",
                "result": result.model_dump(mode="json"),
                "final": True,
            }
        )

    try:
        if not payload.get("isolated"):
            provider = setup_instrumentation(
                project_name=payload["project_name"],
                endpoint=payload.get("otel_endpoint"),
                framework=framework_for_agent(payload["cell"]["harness"]),
                batch=True,
                extra_resource_attributes=(
                    ("a2e.campaign_id", payload["cell"]["campaign_id"]),
                    ("a2e.cell_id", payload["cell"]["cell_id"]),
                ),
            )
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def read_commands() -> None:
            while True:
                message = command_queue.get()
                loop.call_soon_threadsafe(inbox.put_nowait, message)
                if message.get("type") == "_stop_reader":
                    return

        command_reader = threading.Thread(
            target=read_commands,
            name=f"a2e-ipc-{payload['cell']['cell_id'][-8:]}",
            daemon=True,
        )
        command_reader.start()
        await send({"type": "worker_ready", "cell_id": payload["cell"]["cell_id"]})
        stopping = False
        while True:
            try:
                message = await asyncio.wait_for(inbox.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if stopping and not trial_tasks and not reply_futures:
                    break
                continue
            kind = message.get("type")
            if kind == "run_trial":
                if stopping:
                    continue
                trial = message["trial"]
                task = asyncio.create_task(run_trial(trial), name=f"trial-{trial['trial_id']}")
                trial_tasks[trial["trial_id"]] = task

                def finished(_task: asyncio.Task[None], trial_id: str = trial["trial_id"]) -> None:
                    trial_tasks.pop(trial_id, None)
                    if not _task.cancelled() and _task.exception() is not None:
                        logger.error("trial task %s crashed: %s", trial_id, _task.exception())

                task.add_done_callback(finished)
            elif kind == "lifecycle_response":
                future = reply_futures.get(message["request_id"])
                if future is not None and not future.done():
                    future.set_result(message)
            elif kind == "cancel":
                stopping = True
                for task in list(trial_tasks.values()):
                    task.cancel()
            elif kind == "shutdown":
                stopping = True
            elif kind == "_stop_reader":
                continue
            else:
                logger.warning("unknown Controller message: %s", kind)
            if stopping and not trial_tasks and not reply_futures:
                break
    finally:
        try:
            command_queue.put({"type": "_stop_reader"})
        except (OSError, ValueError):
            pass
        command_reader = locals().get("command_reader")
        if command_reader is not None:
            command_reader.join(timeout=1)
        if provider is not None:
            provider.force_flush(timeout_millis=10_000)
            provider.shutdown()


def run_cell_worker(payload: dict[str, Any], command_queue: Any, event_queue: Any) -> None:
    """Top-level multiprocessing entry point."""
    try:
        asyncio.run(_run_cell_async(payload, command_queue, event_queue))
    except BaseException as exc:
        event_queue.put(
            {
                "type": "worker_error",
                "cell_id": payload["cell"]["cell_id"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
            }
        )
    finally:
        event_queue.put({"type": "worker_done", "cell_id": payload["cell"]["cell_id"]})
