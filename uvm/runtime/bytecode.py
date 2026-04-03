"""Bytecode definitions for UVM.

Defines the instruction set for the stack-based UVM.
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Optional


class Op(IntEnum):
    """UVM Opcodes."""
    
    # Stack operations
    LOAD_CONST = 0          # Load constant onto stack
    LOAD_VAR = 1            # Load variable onto stack
    STORE_VAR = 2           # Store stack top to variable
    POP = 3                 # Pop stack
    DUP = 4                 # Duplicate stack top
    
    # Arithmetic/Logic
    ADD = 10                # Add top two stack items
    EQ = 11                 # Compare equality
    LT = 12                 # Less than
    GT = 13                 # Greater than
    
    # Control flow
    JUMP = 20               # Unconditional jump
    JUMP_IF_FALSE = 21      # Jump if stack top is falsy
    
    # UVM-specific
    CALL = 30               # Call LM/Agent
    YIELD_CALL = 31         # Yield before calling (for memory management)
    INPUT = 32              # Read user input
    RETURN = 33             # Return from execution
    
    # Block operations
    DO_WHILE_START = 40     # Mark start of DO-WHILE
    DO_WHILE_END = 41       # Check condition, jump back if true
    BREAK = 42              # Break out of current loop
    
    # Agent operations
    AGENT_CREATE = 50       # Create and register agent
    
    # Misc
    NOP = 99                # No operation


@dataclass(frozen=True)
class Instruction:
    """A single UVM instruction."""
    op: Op
    operand: Any = None
    
    def __repr__(self):
        if self.operand is not None:
            return f"{self.op.name}({self.operand!r})"
        return self.op.name


class VMState(IntEnum):
    """UVM execution states."""
    RUNNING = auto()
    YIELDED = auto()        # Paused for external call
    YIELD_CHECKPOINT = auto()  # Engine requested checkpoint
    HALTED = auto()
    ERROR = auto()


# Built-in opcode handlers mapping (populated in vm.py)
_opcode_handlers = {}


def register_opcode_handler(op: Op, handler):
    """Register a handler for an opcode."""
    _opcode_handlers[op] = handler


def get_opcode_handler(op: Op):
    """Get the handler for an opcode."""
    return _opcode_handlers.get(op)
