"""AGENT - Agent class for UVM.

An AGENT wraps an LM with a harness for memory management and
provides a higher-level interface for multi-agent workflows.
"""

from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from .memstr import MemStr, to_memstr
from .harness import HARNESS, DefaultHarness
from .memory_engine import AgentStruct

if TYPE_CHECKING:
    from .context import UVMContext


class AGENT:
    """Agent class wrapping an LM with memory and harness.
    
    Attributes:
        name: Agent identifier
        lm: The underlying LM (by name or object)
        memory: Dictionary of MemStr for stateful storage
        harness: HARNESS for input/output transformation
    """
    
    def __init__(
        self,
        name: str,
        lm: Union[str, "LM"],
        prompt: str,
        harness: Optional[HARNESS] = None,
        context: Optional["UVMContext"] = None,
        memory_engine: Optional[Any] = None
    ):
        """Initialize AGENT.
        
        Args:
            name: Unique identifier for this agent
            lm: LM instance or name of registered LM
            prompt: System prompt/context for this agent
            harness: Optional HARNESS (defaults to identity)
            context: UVMContext for registration
            memory_engine: MemoryEngine for lifecycle tracking
        """
        self.name = name
        self.lm = lm  # Can be LM object or string name
        self.prompt = prompt
        self.harness = harness or DefaultHarness()
        self.memory: Dict[str, MemStr] = {}
        self._memory_engine = memory_engine
        self._context = context
        
        # Register in context
        if context:
            context.register_agent(name, self)
        
        # Notify memory engine
        if memory_engine:
            lm_ref = lm if isinstance(lm, str) else getattr(lm, 'name', str(id(lm)))
            agent_struct = AgentStruct(
                name=name,
                lm_ref=lm_ref,
                prompt_preview=prompt[:100],
                harness_type=type(self.harness).__name__,
                memory_schema={}
            )
            memory_engine.on_agent_register(agent_struct)
    
    def call(self, *args) -> MemStr:
        """Call the agent with arguments.
        
        Flow:
        1. Combine prompt with args
        2. Run harness.on_update
        3. Call LM
        4. Run harness.on_return
        5. Return result
        
        Args:
            *args: Variable arguments to pass to LM
        
        Returns:
            MemStr with the response.
        """
        # Get the actual LM object
        lm_obj = self._resolve_lm()
        
        # Combine prompt with args
        inputs = [self.prompt] + list(args) if self.prompt else list(args)
        
        # Convert to MemStr
        inputs = [to_memstr(i, memory_engine=self._memory_engine, context="agent_input") 
                  for i in inputs]
        
        # Apply harness on_update
        processed = self.harness.on_update(self, inputs)
        if not isinstance(processed, (list, tuple)):
            processed = [processed]
        
        # Call LM
        result = lm_obj(*processed)
        
        # Apply harness on_return
        result = self.harness.on_return(self, result)
        
        # Ensure MemStr
        if not isinstance(result, MemStr):
            result = to_memstr(result, memory_engine=self._memory_engine, context="agent_output")
        
        return result
    
    def _resolve_lm(self) -> Any:
        """Resolve LM reference to actual LM object."""
        if isinstance(self.lm, str):
            if self._context:
                return self._context.get_lm(self.lm)
            raise RuntimeError(f"Cannot resolve LM name '{self.lm}' without context")
        return self.lm
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent memory.
        
        Args:
            key: Memory key
            value: Value to store (converted to MemStr)
        """
        # Release old value if exists
        if key in self.memory:
            self.memory[key]._release()
        
        # Store new value
        ms = to_memstr(value, memory_engine=self._memory_engine, context=f"agent.{self.name}.memory")
        ms._retain()
        self.memory[key] = ms
    
    def get_memory(self, key: str) -> Optional[MemStr]:
        """Get value from agent memory."""
        return self.memory.get(key)
    
    def clear_memory(self) -> None:
        """Clear all agent memory."""
        for ms in self.memory.values():
            ms._release()
        self.memory.clear()
    
    def __call__(self, *args) -> MemStr:
        """Allow AGENT to be called directly."""
        return self.call(*args)
    
    def __repr__(self):
        lm_name = self.lm if isinstance(self.lm, str) else getattr(self.lm, 'name', 'unnamed')
        return f"AGENT(name={self.name!r}, lm={lm_name!r})"
