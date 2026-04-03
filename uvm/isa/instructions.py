"""ISA Instructions for UVM.

High-level instructions that operate on the current UVMContext.
These provide a convenient API for writing agentic workflows.
"""

from typing import Any, Optional

from ..core.context import get_context, set_context, UVMContext
from ..core.memstr import MemStr, to_memstr
from ..core.memory_engine import CallEvent
from ..runtime import parse, compile_ast, UVM


def _ensure_context():
    """Ensure a default context exists and is active."""
    try:
        return get_context()
    except RuntimeError:
        # Import here to avoid circular import
        from .. import _default_context, UVMContext
        ctx = _default_context
        if ctx is None:
            ctx = UVMContext()
        set_context(ctx)
        return ctx


def CALL(name: str, *args) -> MemStr:
    """Call an LM or Agent by name.
    
    Args:
        name: Name of registered LM or Agent
        *args: Arguments to pass
    
    Returns:
        MemStr with the response.
    
    Raises:
        KeyError: If name not found in registry
    """
    ctx = _ensure_context()
    
    # Resolve name
    obj = ctx.get(name)
    
    # Convert args to MemStr
    mem_args = [to_memstr(arg, ctx.memory_engine, "call_arg") for arg in args]
    
    # Notify memory engine
    handle = None
    if ctx.memory_engine:
        event = CallEvent(
            caller=None,
            callee=name,
            args_summary=tuple(type(a).__name__ for a in mem_args),
            token_estimate=sum(len(str(a)) for a in mem_args) // 4
        )
        handle = ctx.memory_engine.on_call_enter(event)
    
    try:
        # Call the object
        if hasattr(obj, 'call'):
            result = obj.call(*mem_args)
        elif callable(obj):
            result = obj(*mem_args)
        else:
            raise TypeError(f"Object '{name}' is not callable")
        
        # Ensure MemStr
        if not isinstance(result, MemStr):
            result = to_memstr(result, ctx.memory_engine, "call_result")
        
    finally:
        # Notify memory engine of completion
        if ctx.memory_engine and handle is not None:
            ctx.memory_engine.on_call_exit(event, handle, result)
    
    return result


def AGENT(name: str, lm: str, prompt: str) -> None:
    """Create and register an Agent.
    
    Args:
        name: Unique identifier for the agent
        lm: Name of registered LM to use
        prompt: System prompt for the agent
    """
    from ..core.agent import AGENT as AgentClass
    
    ctx = _ensure_context()
    
    # Get LM
    lm_obj = ctx.get_lm(lm)
    
    # Create agent (auto-registers)
    AgentClass(
        name=name,
        lm=lm_obj,
        prompt=prompt,
        context=ctx,
        memory_engine=ctx.memory_engine
    )


def INPUT() -> MemStr:
    """Read user input.
    
    Returns:
        MemStr with user input.
    """
    ctx = _ensure_context()
    result = input()
    return to_memstr(result, ctx.memory_engine, "input")


def RETURN(value: Any = None) -> None:
    """Return from execution.
    
    Note: In USL, this is handled by the VM. In Python, this can be
    used in harnesses or called directly to return from a block.
    
    Args:
        value: Value to return
    """
    # When used directly in Python, this is essentially a no-op
    # or could raise a control flow exception
    return value


def EXEC(plan: str, initial_vars: Optional[dict] = None) -> Any:
    """Execute USL plan.
    
    Parses the USL string, compiles to bytecode, and executes in the UVM.
    
    Args:
        plan: USL source code string
        initial_vars: Optional initial variable values
    
    Returns:
        Result of the USL execution.
    """
    ctx = _ensure_context()
    
    # Parse USL to AST
    ast = parse(plan)
    
    # Compile to bytecode
    bytecode = compile_ast(ast)
    
    # Convert initial_vars to MemStr
    if initial_vars:
        initial_vars = {
            k: to_memstr(v, ctx.memory_engine, "exec_init") 
            for k, v in initial_vars.items()
        }
    
    # Create and run VM
    vm = UVM(bytecode, ctx, initial_vars)
    result = vm.run_to_completion()
    
    return result


# Control flow helpers (these are no-ops in Python, used in USL)

def DO(body):
    """Marker for DO...WHILE loops.
    
    In USL, this is compiled to bytecode. In Python,
    this is just a pass-through for documentation.
    """
    return body


def WHILE(condition):
    """Marker for DO...WHILE condition.
    
    In USL, this is compiled to bytecode. In Python,
    this returns the condition.
    """
    return condition


def IF(condition):
    """Marker for IF statement.
    
    In USL, this is compiled to bytecode. In Python,
    this returns the condition.
    """
    return condition


def BREAK():
    """Break from loop.
    
    In USL, this compiles to BREAK opcode. In Python,
    this raises an exception for use in control flow.
    """
    raise USLBreak()


class USLBreak(Exception):
    """Exception used for BREAK in USL simulation."""
    pass


# Convenience function to check if a name is registered

def is_registered(name: str) -> bool:
    """Check if a name is registered in the current context."""
    try:
        ctx = _ensure_context()
        return ctx.has(name)
    except RuntimeError:
        return False


# Functions for direct registry manipulation (from demo.py interface)

def register(name: str, obj: Any) -> None:
    """Manually register an LM or Agent.
    
    Args:
        name: Identifier
        obj: LM or AGENT instance
    """
    ctx = _ensure_context()
    if hasattr(obj, 'lm'):  # It's an agent
        ctx.register_agent(name, obj)
    else:  # Assume it's an LM
        ctx.register_lm(name, obj)


def delete(name: str) -> None:
    """Delete a registered LM or Agent.
    
    Args:
        name: Identifier to delete
    """
    ctx = _ensure_context()
    ctx.delete(name)
