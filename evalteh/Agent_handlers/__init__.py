import importlib
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import BaseModel


class PoolEnum(StrEnum):
    NONE = "None"
    ANCHOR = "Anchor"
    STEEL = "Steel"
    BROWSERBASE = "BrowserBase"
    CAMOUFOX = "Camoufox"


class Proxy(BaseModel):
    server: str
    username: str
    password: str


def fetch_handler(key: str) -> tuple[type, type]:
    """
    Import specific module based on key and return input and handler types
    """
    if key not in HANDLERS_DICT:
        raise ValueError(f"Unknown handler key: {key}")

    handler = HANDLERS_DICT[key]
    module = importlib.import_module(f"{__package__}.{handler.module_name}")

    input_type = getattr(module, handler.input_name)
    handler_type = getattr(module, handler.handler_name)

    return input_type, handler_type


class HandlerTuple(NamedTuple):
    module_name: str
    input_name: str
    handler_name: str


HANDLERS_DICT = {
    "LumiTeh": HandlerTuple("falco", "FalcoInput", "FalcoBench"),
    "BrowserUse": HandlerTuple("browseruse", "BrowserUseInput", "BrowserUseBench"),
    "BrowserUseAPI": HandlerTuple("browseruse_api", "BrowserUseAPIInput", "BrowserUseAPIBench"),
    "Convergence": HandlerTuple("convergence", "ConvergenceInput", "ConvergenceBench"),
}


def trim_image_messages(input_content: list[dict[Any, Any]]) -> None:
    """
    Remove image URLs from message history for benchmarking purposes.
    """
    for msg in input_content:
        if "content" in msg and isinstance(msg["content"], list):
            for submsg in msg["content"]:  # type: ignore
                if "type" in submsg and submsg["type"] == "image_url" and "image_url" in submsg:
                    submsg["image_url"] = "benchmark: removed"
