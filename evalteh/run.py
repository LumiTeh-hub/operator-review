from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import io
import logging
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import cloudpickle  # type: ignore[reportMissingTypeStubs]
import pebble
from loguru import logger as loguru_logger
from pydantic import BaseModel
from typing_extensions import Self

from lumiteh.utils.webp_replay import ScreenshotReplay
from eval.agent_handlers import fetch_handler
from eval.data.load_data import BenchmarkTask
from eval.evaluators import EVALUATORS_DICT, fetch_evaluator
from eval.evaluators.evaluator import Evaluator
from eval.task_types import AgentBenchmark, AgentOut, AgentParams, LoggingSink, TaskResult

from dotenv import load_dotenv
load_dotenv()


class TaskSet(BaseModel):
    name: str
    start: int | None = None
    end: int | None = None


class RunParameters(BaseModel):
    n_jobs: int
    tries_per_task: int
    task_set: TaskSet
    max_task_duration_in_s: float = 5 * 60
    evaluator: Evaluator | None = None
    experiment_path: Path | str = ""
    capture_logging: bool = True


class InRunParameters(BaseModel):
    class Config:
        frozen: bool = True

    run_id: int
    evaluator: Evaluator | None = None
    experiment_path: Path | str = ""
    capture_logging: bool = True


TaskSuccessResult = tuple[BenchmarkTask, AgentOut, TaskResult]


@dataclass
class TaskErrorResult:
    task: BenchmarkTask
    run_params: InRunParameters
    logs: dict[str, str]
    experiment_path: str | Path
    exception: Exception | None = None
    traceback_str: str | None = None
    logged: bool = False

    def log(self) -> Self:
        if self.logged:
            return self

        task_res = TaskResult(
            success=False,
            run_id=self.run_params.run_id,
            eval=None,
            duration_in_s=-1,
            agent_answer=f"Task failed {'due to ' + str(self.exception) if self.exception is not None else ''} {self.traceback_str}",
            task=self.task,
            steps=[],
            logs=self.logs,
            screenshots=ScreenshotReplay.from_base64([]),
        )

        save_task(self.experiment_path, task_res)
        self.logged = True
        return self


class BenchmarkExecutionResult:
    def __init__(self, success: bool, data: TaskSuccessResult[Any] | TaskErrorResult):
        self.success: bool = success
        self.data: TaskSuccessResult[Any] | TaskErrorResult = data

    @classmethod
    def successful(cls, data: TaskSuccessResult[Any]) -> Self:
        return cls(True, data)

    @classmethod
    def failure(cls, error_result: TaskErrorResult) -> Self:
        return cls(False, error_result)


def setup_logging(log_stream: io.StringIO) -> None:
    logging.getLogger().setLevel(logging.INFO)
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(stream_handler)


def sync_wrapper(async_func: Callable[..., Any], *args: tuple[Any, ...], **kwargs: dict[str, Any]):
    try:
        return asyncio.run(async_func(*args, **kwargs))
    except RuntimeError as e:
        if "There is no current event loop" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_func(*args, **kwargs))
        else:
            raise


async def run_agent(
    agent_bench: AgentBenchmark[AgentParams, AgentOut],
    task: BenchmarkTask,
    inrun_params: InRunParameters,
) -> bytes | TaskErrorResult:
    log_capture = io.StringIO()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    sink = LoggingSink()

    if inrun_params.capture_logging:
        loguru_logger.remove()
        _ = loguru_logger.add(sink, level="DEBUG")

        setup_logging(log_capture)
    else:
        stdout_capture = sys.stdout
        stderr_capture = sys.stderr

    def get_logs() -> dict[str, str]:
        if not inrun_params.capture_logging:
            return {}
        assert isinstance(stderr_capture, io.StringIO) and isinstance(stdout_capture, io.StringIO)
        logs: dict[str, str] = {}
        logs["stdout"] = stdout_capture.getvalue()
        logs["stderr"] = stderr_capture.getvalue()
        logs["logging"] = log_capture.getvalue()
        logs["loguru"] = "\n".join(sink.messages)
        return logs

    try:
        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            run = await agent_bench.run_agent(task)
            out = await agent_bench.process_output(task, run)
            out.run_id = inrun_params.run_id

            if inrun_params.evaluator is not None:
                out.eval = await inrun_params.evaluator.eval(
                    out.agent_answer, task.question, out.screenshots.b64_screenshots
                )

        if inrun_params.capture_logging:
            out.logs = get_logs()

        save_task(inrun_params.experiment_path, out)
        return cloudpickle.dumps((task, run, out))

    except Exception as e:
        logging.error(f"{e}: {traceback.format_exc()}")
        return TaskErrorResult(
            task,
            inrun_params,
            get_logs(),
            inrun_params.experiment_path,
            exception=e,
            traceback_str=traceback.format_exc(),
        ).log()


def compute_tasks(
    agent_bench: AgentBenchmark[AgentParams, AgentOut], run_parameters: RunParameters
) -> list[BenchmarkExecutionResult]:
    try:
        task_class = BenchmarkTask.registry[run_parameters.task_set.name]
    except KeyError:
        raise ValueError(f"Invalid task set {run_parameters.task_set}, available: {BenchmarkTask.registry.keys()}")

    tasks = task_class.read_tasks()
    task_slice = slice(run_parameters.task_set.start, run_parameters.task_set.end)
    tasks = tasks[task_slice]

    futures: list[tuple[BenchmarkTask, InRunParameters, pebble.ProcessFuture]] = []
    gathered_outputs: list[bytes | TaskErrorResult] = []

    with pebble.ProcessPool(max_workers=run_parameters.n_jobs, max_tasks=1) as pool:
        for task in tasks:
            for run_id in range(run_parameters.tries_per_task):
                run_params = InRunParameters(
                    run_id=run_id,
                    evaluator=run_parameters.evaluator,
                    experiment_path=run_parameters.experiment_path,
                    capture_logging=run_parameters.capture_logging,
                )

                wrapped_task = functools.partial(sync_wrapper, run_agent, agent_bench, task, run_params)
                future = pool.schedule(wrapped_task, timeout=run_parameters.max_task_duration_in_s)
                futures.append((task, run_params, future))

        try:
            for task, run_params, future in futures:
                try:
                    result = future.result()
                    assert isinstance(result, (bytes, TaskErrorResult))
                    gathered_outputs.append(result)
                except Exception as e:
                    gathered_outputs.append(
                        TaskErrorResult(
                            task,
                            run_params,
                            {},
                            run_params.experiment_path,
                            exception=e,
                            traceback_str=traceback.format_exc(),
                        ).log()
                    )

        except KeyboardInterrupt:
            pool.stop()
            pool.join()
        finally:
            pool.close()
            pool.join()

    final_outs: list[BenchmarkExecutionResult] = []
    for out in gathered_outputs:
        if isinstance(out, bytes):
            try:
                task_outputs: TaskSuccessResult = cloudpickle.loads(out)
                final_outs.append(BenchmarkExecutionResult.successful(task_outputs))
            except Exception:
                raise ValueError(f"Could not read bytes from task return: {traceback.format_exc()}")
        else:
            final_outs.append(BenchmarkExecutionResult.failure(out))

    return final_outs


def save_task(root_path: str | Path, task_res: TaskResult):
    path = Path(root_path) if not isinstance(root_path, Path) else root_path
    path = path / f"{task_res.task_website}_{task_res.task_id}" / str(task_res.run_id)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "results.json", "w") as f:
        _ = f.write(task_res.model_dump_json(indent=2))

    with open(path / "results_no_screenshot.json", "w") as f:
        _ = f.write(task_res.model_dump_json(indent=2, exclude={"screenshots"}))

    with open(path / "summary.webp", "wb") as f:
        _ = f.write(task_res.screenshots.summary_webp(start_text=task_res.task.question))


def load_data(input_stream: TextIO | None = None) -> dict[str, Any]:
    stream: TextIO = input_stream if
