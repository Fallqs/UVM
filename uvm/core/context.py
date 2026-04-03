"""UVMContext - Thread-safe execution context for UVM.

Replaces global dicts with an explicit context model that supports
parallelism and memory management.
"""

import threading
from typing import Any, Dict, Optional
from contextvars import ContextVar

from .memory_engine import MemoryEngine, NullMemoryEngine

# Context-local storage for current UVMContext
_current_context: ContextVar[Optional["UVMContext"]] = ContextVar('uvm_context', default=None)


class UVMContext:
    """Execution context for UVM.
    
    Holds registries for LMs and Agents, plus the memory engine.
    Each thread/async context should use its own UVMContext for isolation.
    
    Attributes:
        lm_registry: Dict mapping names to LM instances
        agent_registry: Dict mapping names to AGENT instances
        memory_engine: Pluggable memory management engine
        _lock: Threading lock for registry operations
    """
    
    def __init__(self, memory_engine: Optional[MemoryEngine] = None):
        """Initialize UVMContext.
        
        Args:
            memory_engine: MemoryEngine implementation (defaults to NullMemoryEngine)
        """
        self.lm_registry: Dict[str, Any] = {}
        self.agent_registry: Dict[str, Any] = {}
        self.memory_engine = memory_engine or NullMemoryEngine()
        self._lock = threading.RLock()
        
        # Inject self into engine for cross-referencing
        self._lm_counter = 0
        self._agent_counter = 0
    
    def register_lm(self, name: str, lm: Any) -> None:
        """Register an LM by name.
        
        Args:
            name: Identifier for the LM
            lm: LM instance
        """
        with self._lock:
            self.lm_registry[name] = lm
            self._lm_counter += 1
    
    def register_agent(self, name: str, agent: Any) -> None:
        """Register an AGENT by name.
        
        Args:
            name: Identifier for the agent
            agent: AGENT instance
        """
        with self._lock:
            self.agent_registry[name] = agent
            self._agent_counter += 1
    
    def delete(self, name: str) -> None:
        """Delete an LM or agent by name.
        
        Tries agent_registry first, then lm_registry.
        
        Args:
            name: Name to delete
        
        Raises:
            KeyError: If name not found in either registry
        """
        with self._lock:
            if name in self.agent_registry:
                agent = self.agent_registry[name]
                # Notify memory engine
                self.memory_engine.on_agent_unregister(name)
                # Clear agent memory
                if hasattr(agent, 'clear_memory'):
                    agent.clear_memory()
                del self.agent_registry[name]
                self._agent_counter -= 1
            elif name in self.lm_registry:
                del self.lm_registry[name]
                self._lm_counter -= 1
            else:
                raise KeyError(f"'{name}' not found in registry")
    
    def get_lm(self, name: str) -> Any:
        """Get LM by name.
        
        Args:
            name: LM identifier
        
        Returns:
            LM instance
        
        Raises:
            KeyError: If LM not found
        """
        with self._lock:
            if name not in self.lm_registry:
                raise KeyError(f"LM '{name}' not found")
            return self.lm_registry[name]
    
    def get_agent(self, name: str) -> Any:
        """Get AGENT by name.
        
        Args:
            name: Agent identifier
        
        Returns:
            AGENT instance
        
        Raises:
            KeyError: If agent not found
        """
        with self._lock:
            if name not in self.agent_registry:
                raise KeyError(f"Agent '{name}' not found")
            return self.agent_registry[name]
    
    def get(self, name: str) -> Any:
        """Get LM or AGENT by name.
        
        Tries agent_registry first, then lm_registry.
        
        Args:
            name: Identifier
        
        Returns:
            LM or AGENT instance
        
        Raises:
            KeyError: If not found in either registry
        """
        with self._lock:
            if name in self.agent_registry:
                return self.agent_registry[name]
            if name in self.lm_registry:
                return self.lm_registry[name]
            raise KeyError(f"'{name}' not found in registry")
    
    def has(self, name: str) -> bool:
        """Check if name exists in either registry."""
        with self._lock:
            return name in self.agent_registry or name in self.lm_registry
    
    def list_lms(self) -> list:
        """List all registered LM names."""
        with self._lock:
            return list(self.lm_registry.keys())
    
    def list_agents(self) -> list:
        """List all registered agent names."""
        with self._lock:
            return list(self.agent_registry.keys())
    
    def stats(self) -> dict:
        """Get context statistics."""
        with self._lock:
            return {
                "lm_count": len(self.lm_registry),
                "agent_count": len(self.agent_registry),
                "total_registered_lms": self._lm_counter,
                "total_registered_agents": self._agent_counter,
            }


def get_context() -> UVMContext:
    """Get the current UVMContext.
    
    Returns:
        The current context
    
    Raises:
        RuntimeError: If no context is active
    """
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "No UVMContext is active. "
            "Use 'with uvm_context():' or 'set_context(ctx)'"
        )
    return ctx


def set_context(ctx: UVMContext) -> None:
    """Set the current UVMContext.
    
    Args:
        ctx: Context to activate
    """
    _current_context.set(ctx)


def clear_context() -> None:
    """Clear the current UVMContext."""
    _current_context.set(None)


class uvm_context:
    """Context manager for UVMContext.
    
    Usage:
        with uvm_context() as ctx:
            # use ctx
            CALL("m1", "hello")
    """
    
    def __init__(self, memory_engine: Optional[MemoryEngine] = None):
        self.ctx = UVMContext(memory_engine=memory_engine)
        self._token = None
    
    def __enter__(self) -> UVMContext:
        self._token = _current_context.set(self.ctx)
        return self.ctx
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_context.reset(self._token)
        return False
