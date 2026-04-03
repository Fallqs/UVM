"""ISA - Instruction Set Architecture for UVM."""

from .instructions import (
    CALL,
    AGENT,
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

__all__ = [
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
    "is_registered",
    "USLBreak",
]
