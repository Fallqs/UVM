"""Test fixtures for UVM - Mock LM/AGENT implementations."""

from typing import Any, List, Optional
from uvm.core.memstr import MemStr, to_memstr
from uvm.core.context import UVMContext, set_context
from uvm.core.memory_engine import NullMemoryEngine


class EchoLM:
    """Mock LM that echoes back the input prompt."""

    def __init__(self, prefix: str = "ECHO"):
        self.prefix = prefix
        self.calls: List[tuple] = []

    def __call__(self, *args) -> MemStr:
        prompt = "\n".join(str(a) for a in args)
        self.calls.append(prompt)
        result = f"[{self.prefix}] {prompt}"
        return to_memstr(result)

    def call(self, *args) -> MemStr:
        return self(*args)


class CounterLM:
    """Mock LM that returns incrementing counters."""

    def __init__(self):
        self.count = 0
        self.calls: List[tuple] = []

    def __call__(self, *args) -> MemStr:
        self.count += 1
        prompt = "\n".join(str(a) for a in args)
        self.calls.append(prompt)
        result = f"[COUNT:{self.count}] {prompt}"
        return to_memstr(result)

    def call(self, *args) -> MemStr:
        return self(*args)


class JudgeLM:
    """Mock LM that returns YES/NO based on keyword matching."""

    def __init__(self, yes_keyword: str = "done", no_keyword: str = "pending"):
        self.yes_keyword = yes_keyword
        self.no_keyword = no_keyword
        self.calls: List[tuple] = []

    def __call__(self, *args) -> MemStr:
        prompt = "\n".join(str(a) for a in args)
        self.calls.append(prompt)
        if self.yes_keyword in prompt.lower():
            return to_memstr("YES")
        if self.no_keyword in prompt.lower():
            return to_memstr("NO")
        return to_memstr("MAYBE")

    def call(self, *args) -> MemStr:
        return self(*args)


class FixedLM:
    """Mock LM that always returns a fixed response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: List[tuple] = []

    def __call__(self, *args) -> MemStr:
        prompt = "\n".join(str(a) for a in args)
        self.calls.append(prompt)
        return to_memstr(self.response)

    def call(self, *args) -> MemStr:
        return self(*args)


def create_test_context() -> UVMContext:
    """Create a fresh UVMContext with null memory engine for testing."""
    ctx = UVMContext(memory_engine=NullMemoryEngine())
    set_context(ctx)
    return ctx
