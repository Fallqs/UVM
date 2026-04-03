"""UVM - Uni-Virtual Machine.

Atomic multi-agent orchestration engine with bytecode VM,
memory management, and USL (UVM Specific Language) support.
"""

# Core exports
from .core import (
    # Context
    UVMContext,
    get_context,
    set_context,
    clear_context,
    uvm_context,
    
    # Memory
    MemStr,
    to_memstr,
    MemoryEngine,
    AgentStruct,
    CallEvent,
    NullMemoryEngine,
    LoggingMemoryEngine,
    CountingMemoryEngine,
    
    # Config
    load_config,
    
    # Actors
    LM,
    AGENT,
    HARNESS,
    DefaultHarness,
    MemoryAugmentedHarness,
    LoggingHarness,
)

# ISA exports
from .isa import (
    CALL,
    AGENT as AGENT_ISA,  # Renamed to avoid conflict with core.AGENT
    INPUT,
    RETURN,
    EXEC,
    DO,
    WHILE,
    IF,
    BREAK,
    register,
    delete,
    is_registered,
    USLBreak,
)

# Runtime exports
from .runtime import (
    UVM as UVMRuntime,  # Renamed to avoid conflict with package name
    parse,
    compile_ast,
    Op,
    Instruction,
    VMState,
)

# Default context for lazy initialization
_default_context = None

def _ensure_context():
    """Ensure a default context exists."""
    global _default_context
    try:
        return get_context()
    except RuntimeError:
        if _default_context is None:
            _default_context = UVMContext()
        # Always set as current context
        set_context(_default_context)
        return _default_context

# For demo.py compatibility, we need direct exports
__all__ = [
    # Context
    "UVMContext",
    "get_context",
    "set_context",
    "clear_context",
    "uvm_context",
    
    # Core types
    "MemStr",
    "LM",
    "AGENT",
    "HARNESS",
    "DefaultHarness",
    "MemoryAugmentedHarness",
    "LoggingHarness",
    
    # Memory Engine
    "MemoryEngine",
    "AgentStruct",
    "CallEvent",
    "NullMemoryEngine",
    "LoggingMemoryEngine",
    "CountingMemoryEngine",
    
    # ISA
    "CALL",
    "AGENT",
    "INPUT",
    "RETURN",
    "EXEC",
    "DO",
    "WHILE",
    "IF",
    "BREAK",
    "register",
    "delete",
    "USLBreak",
    
    # Runtime
    "UVM",
    "parse",
    "compile_ast",
    "Op",
    "Instruction",
    "VMState",
]

# For convenience, bind ISA AGENT to the function
AGENT = AGENT_ISA

# For convenience, bind UVM to the runtime
UVM = UVMRuntime

# Wrapper functions for demo compatibility
def register(name: str, obj):
    """Register an LM or Agent in the default context."""
    ctx = _ensure_context()
    from .isa import register as isa_register
    return isa_register(name, obj)

def delete(name: str):
    """Delete an LM or Agent from the default context."""
    ctx = _ensure_context()
    from .isa import delete as isa_delete
    return isa_delete(name)
