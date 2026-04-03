"""Memory Engine API for UVM.

Third-party memory management engines implement the MemoryEngine ABC
to integrate with UVM for KV-cache management, hierarchical memory,
and distributed state stores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memstr import MemStr


@dataclass(frozen=True)
class AgentStruct:
    """Structural snapshot of an AGENT at creation time."""
    name: str
    lm_ref: str           # name of the underlying LM
    prompt_preview: str   # first N chars of system prompt (for identification)
    harness_type: str     # class name of harness
    memory_schema: dict   # keys of memory dict, types as strings


@dataclass(frozen=True)
class CallEvent:
    """A CALL instruction event."""
    caller: Optional[str]   # agent name, or None if direct LM call
    callee: str             # LM or agent name being invoked
    args_summary: tuple     # type names of args (not full content to avoid copy)
    token_estimate: int     # rough estimate for budgeting


class MemoryEngine(ABC):
    """Pluggable memory management interface.
    
    Third parties implement this to integrate UVM with their KV-cache managers,
    hierarchical memory systems, or distributed state stores.
    """
    
    @abstractmethod
    def on_agent_register(self, agent_struct: AgentStruct) -> None:
        """Called when AGENT() creates and registers a new agent."""
        pass
    
    @abstractmethod
    def on_agent_unregister(self, name: str) -> None:
        """Called when an agent is deleted from context."""
        pass
    
    @abstractmethod
    def on_call_enter(self, event: CallEvent) -> Optional[Any]:
        """Called before CALL executes. 
        
        Returns: optional context handle for on_call_exit.
        """
        pass
    
    @abstractmethod
    def on_call_exit(self, event: CallEvent, handle: Optional[Any], result: "MemStr") -> None:
        """Called after CALL returns with result."""
        pass
    
    @abstractmethod
    def on_memstr_create(self, ms: "MemStr", context: str) -> None:
        """Called when MemStr is instantiated.
        
        Args:
            ms: The MemStr being created
            context: Where it's created ('stack', 'agent.memory', 'locals', etc)
        """
        pass
    
    @abstractmethod
    def on_memstr_drop(self, ms: "MemStr") -> None:
        """Called when MemStr refcount hits 0."""
        pass
    
    @abstractmethod
    def checkpoint_request(self) -> bool:
        """Called by UVM runtime when memory pressure is high.
        
        Returns: True if the engine wants the UVM to yield/hibernate.
        """
        pass


class NullMemoryEngine(MemoryEngine):
    """Default no-op engine."""
    
    def on_agent_register(self, agent_struct: AgentStruct) -> None:
        pass
    
    def on_agent_unregister(self, name: str) -> None:
        pass
    
    def on_call_enter(self, event: CallEvent) -> Optional[Any]:
        return None
    
    def on_call_exit(self, event: CallEvent, handle: Optional[Any], result: "MemStr") -> None:
        pass
    
    def on_memstr_create(self, ms: "MemStr", context: str) -> None:
        pass
    
    def on_memstr_drop(self, ms: "MemStr") -> None:
        pass
    
    def checkpoint_request(self) -> bool:
        return False


class LoggingMemoryEngine(MemoryEngine):
    """Debug engine that prints all events."""
    
    def __init__(self, logger=None):
        import logging
        self.logger = logger or logging.getLogger(__name__)
    
    def on_agent_register(self, agent_struct: AgentStruct) -> None:
        self.logger.debug(f"[MEM] Agent registered: {agent_struct}")
    
    def on_agent_unregister(self, name: str) -> None:
        self.logger.debug(f"[MEM] Agent unregistered: {name}")
    
    def on_call_enter(self, event: CallEvent) -> Optional[Any]:
        self.logger.debug(f"[MEM] CALL enter: {event}")
        return None
    
    def on_call_exit(self, event: CallEvent, handle: Optional[Any], result: "MemStr") -> None:
        self.logger.debug(f"[MEM] CALL exit: {event.callee} -> {len(result)} chars")
    
    def on_memstr_create(self, ms: "MemStr", context: str) -> None:
        self.logger.debug(f"[MEM] MemStr created in {context}: {len(ms)} chars")
    
    def on_memstr_drop(self, ms: "MemStr") -> None:
        self.logger.debug(f"[MEM] MemStr dropped: {len(ms)} chars")
    
    def checkpoint_request(self) -> bool:
        self.logger.debug("[MEM] Checkpoint request (denied)")
        return False


class CountingMemoryEngine(MemoryEngine):
    """Tracks stats for testing."""
    
    def __init__(self):
        self.agent_registers = 0
        self.agent_unregisters = 0
        self.call_enters = 0
        self.call_exits = 0
        self.memstr_creates = 0
        self.memstr_drops = 0
        self.checkpoint_requests = 0
    
    def on_agent_register(self, agent_struct: AgentStruct) -> None:
        self.agent_registers += 1
    
    def on_agent_unregister(self, name: str) -> None:
        self.agent_unregisters += 1
    
    def on_call_enter(self, event: CallEvent) -> Optional[Any]:
        self.call_enters += 1
        return None
    
    def on_call_exit(self, event: CallEvent, handle: Optional[Any], result: "MemStr") -> None:
        self.call_exits += 1
    
    def on_memstr_create(self, ms: "MemStr", context: str) -> None:
        self.memstr_creates += 1
    
    def on_memstr_drop(self, ms: "MemStr") -> None:
        self.memstr_drops += 1
    
    def checkpoint_request(self) -> bool:
        self.checkpoint_requests += 1
        return False
    
    def stats(self) -> dict:
        return {
            "agent_registers": self.agent_registers,
            "agent_unregisters": self.agent_unregisters,
            "call_enters": self.call_enters,
            "call_exits": self.call_exits,
            "memstr_creates": self.memstr_creates,
            "memstr_drops": self.memstr_drops,
            "checkpoint_requests": self.checkpoint_requests,
        }
