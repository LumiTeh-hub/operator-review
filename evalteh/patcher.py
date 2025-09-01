from __future__ import annotations

import asyncio
import functools
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

from pydantic import BaseModel

from lumiteh.llms.logging import recover_args


class CantPatchFunctionError(Exception):
    pass


class CantDumpArgumentError(Exception):
    pass


@dataclass
class FunctionLog:
    start_time: float
    end_time: float
    input_data: Any
    output_data: Any

    @cached_property
    def duration_in_s(self) -> float:
        return self.end_time - self.start_time


class AgentPatcher:
    """
    Patched methods of a LumiTeh agent to monitor execution and track inputs/outputs.
    Designed for use on one class at a time.
    """

    def __init__(self):
        self.logged_data: dict[str, list[FunctionLog]] = defaultdict(list)
        self.prepatch_methods: dict[str, Callable[..., Any]] = {}

    @staticmethod
    def _dump_args(to_dump: Any) -> Any:
        def dump_default(value: Any) -> dict[str, Any] | str:
            if isinstance(value, BaseModel):
                return value.model_dump()
            return str(value)

        return json.dumps(to_dump, default=dump_default)

    def _patch_function(
        self,
        target_class: object,
        func_name: str,
        patching_function: Callable[..., Callable[..., Any]],
    ) -> None:
        func: Callable[..., Any] = getattr(target_class, func_name)

        if func.__qualname__ in self.prepatch_methods:
            raise CantPatchFunctionError(f"Function {func.__qualname__} already patched")

        if func_name == "__call__":
            original_unbound = target_class.__class__.__call__
            patched = patching_function(original_unbound)

            class _(type(target_class)):
                def __call__(self_cls, *args, **kwargs):  # type: ignore
                    return patched(self_cls, *args, **kwargs)

            target_class.__class__ = _
        else:
            patched = patching_function(func)
            try:
                setattr(target_class, func_name, patched)
            except ValueError:
                try:
                    import pydantic
                    if isinstance(target_class, pydantic.BaseModel):
                        target_class.__dict__[func_name] = patched
                except ImportError:
                    raise CantPatchFunctionError(f"Could not setattr {func_name}")
            except Exception as e:
                raise CantPatchFunctionError(f"Could not setattr {func_name}: {e}")

        self.prepatch_methods[func.__qualname__] = patched

    def log(
        self,
        target_class: object,
        methods_to_log: list[str],
        pre_callback: Callable[..., None] | None = None,
        post_callback: Callable[..., None] | None = None,
    ) -> None:
        """Wrap methods to record input/output and execution time."""

        def logging_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                params = recover_args(func, args, kwargs)
                input_params = AgentPatcher._dump_args(params)

                if pre_callback:
                    pre_callback(params)

                if asyncio.iscoroutinefunction(func):
                    async def async_wrapper():
                        result = await func(*args, **kwargs)
                        end = time.time()
                        out_params = AgentPatcher._dump_args(result)
                        self.logged_data[func.__qualname__].append(
                            FunctionLog(start_time=start, end_time=end, input_data=input_params, output_data=out_params)
                        )
                        if post_callback:
                            post_callback(input_params, out_params)
                        return result
                    return async_wrapper()

                result = func(*args, **kwargs)
                end = time.time()
                out_params = AgentPatcher._dump_args(result)
                self.logged_data[func.__qualname__].append(
                    FunctionLog(start_time=start, end_time=end, input_data=input_params, output_data=out_params)
                )
                if post_callback:
                    post_callback(input_params, out_params)
                return result

            return wrapper

        for func_name in methods_to_log:
            self._patch_function(target_class, func_name, logging_decorator)

    def find_encompassed_events(self, container_key: str) -> list[tuple[FunctionLog, dict[str, list[FunctionLog]]]]:
        """
        For each event in container_key, find all events in other keys that are fully encompassed by it.
        """
        results: list[tuple[FunctionLog, dict[str, list[FunctionLog]]]] = []

        for container_event in self.logged_data[container_key]:
            encompassed: dict[str, list[FunctionLog]] = {}
            for key in self.logged_data:
                if key != container_key:
                    contained_events = [
                        e for e in self.logged_data[key]
                        if container_event.start_time <= e.start_time <= e.end_time <= container_event.end_time
                    ]
                    if contained_events:
                        encompassed[key] = contained_events
            results.append((container_event, encompassed))

        return results
