"""HARNESS - Memory control module for AGENT.

The harness provides hooks for intercepting agent inputs and outputs,
enabling dynamic memory management and transformation.
"""

from typing import Callable, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .agent import AGENT


@dataclass
class HARNESS:
    """Memory control module that updates memory on the fly.
    
    The harness intercepts agent calls to:
    - Transform inputs before sending to LM
    - Process/transform outputs before returning
    - Update agent memory state
    
    Attributes:
        on_update: Called with (agent, args) before LM call.
                   Returns the transformed inputs for the LM.
        on_return: Called with (agent, result) after LM returns.
                   Returns the final result.
    """
    on_update: Callable[["AGENT", Any], Any] = lambda _, x: x
    on_return: Callable[["AGENT", Any], Any] = lambda _, x: x
    
    def __call__(self, agent: "AGENT", update_or_return: str, value: Any) -> Any:
        """Call the appropriate hook.
        
        Args:
            agent: The agent being processed
            update_or_return: Either 'update' (on_update) or 'return' (on_return)
            value: The value to process
        
        Returns:
            The processed value.
        """
        if update_or_return == 'update':
            return self.on_update(agent, value)
        elif update_or_return == 'return':
            return self.on_return(agent, value)
        else:
            raise ValueError(f"Unknown hook: {update_or_return}")


def DefaultHarness() -> HARNESS:
    """Create a default harness with identity transformations."""
    return HARNESS()


def MemoryAugmentedHarness(
    memory_key: str = "history",
    max_history: int = 10,
    format_history: Callable[[list], str] = None
) -> HARNESS:
    """Create a harness that maintains conversation history.
    
    Args:
        memory_key: Key in agent.memory to store history
        max_history: Maximum number of turns to keep
        format_history: Function to format history list into string
    """
    def on_update(agent, args):
        # Get history from agent memory
        if memory_key not in agent.memory:
            agent.memory[memory_key] = []
        
        history = agent.memory[memory_key]
        
        # Current input
        current_input = args if isinstance(args, str) else str(args)
        
        # Format with history if formatter provided
        if format_history:
            augmented = format_history(history) + "\nUser: " + current_input
        else:
            if history:
                hist_str = "\n".join([f"User: {h['input']}\nAgent: {h['output']}" 
                                      for h in history[-max_history:]])
                augmented = f"Previous conversation:\n{hist_str}\n\nUser: {current_input}"
            else:
                augmented = current_input
        
        return augmented
    
    def on_return(agent, result):
        # Store in history
        if memory_key not in agent.memory:
            agent.memory[memory_key] = []
        
        # Get the last input (hacky but works for simple cases)
        # In real implementation, this would need better tracking
        agent.memory[memory_key].append({"input": "", "output": str(result)})
        
        # Trim history
        if len(agent.memory[memory_key]) > max_history:
            agent.memory[memory_key] = agent.memory[memory_key][-max_history:]
        
        return result
    
    return HARNESS(on_update=on_update, on_return=on_return)


def LoggingHarness(logger=None) -> HARNESS:
    """Create a harness that logs all inputs and outputs."""
    import logging
    log = logger or logging.getLogger(__name__)
    
    def on_update(agent, args):
        log.info(f"[{agent.name}] Input: {args}")
        return args
    
    def on_return(agent, result):
        log.info(f"[{agent.name}] Output: {result}")
        return result
    
    return HARNESS(on_update=on_update, on_return=on_return)
