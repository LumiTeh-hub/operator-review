from pydantic import BaseModel
from typing_extensions import override

from lumiTeh.utils.webp_replay import ScreenshotReplay
from lumiTeh_eval.data.load_data import BenchmarkTask
from lumiTeh_eval.task_types import AgentBenchmark, TaskResult


class MockInput(BaseModel):
    a: int
    b: bool


class MockOutput(BaseModel):
    s: str


class MockBench(AgentBenchmark[MockInput, MockOutput]):
    def __init__(self, params: MockInput):
        super().__init__(params)

    @override
    async def run_agent(self, task: BenchmarkTask) -> MockOutput:
        # Simply returns the string representation of 'a'
        return MockOutput(s=str(self.params.a))

    @override
    async def process_output(self, task: BenchmarkTask, out: MockOutput) -> TaskResult:
        # Returns a TaskResult with empty steps and no screenshots
        return TaskResult(
            success=False,
            duration_in_s=0,
            agent_answer=out.s,
            task=task,
            steps=[],
            screenshots=ScreenshotReplay.from_base64([]),
        )
