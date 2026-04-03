"""MemStr - Memory-managed string for UVM.

MemStr behaves like a Python str but integrates with the MemoryEngine
for lifecycle tracking and tiered memory management.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine


class MemStr(str):
    """A string subclass with memory management hooks.
    
    MemStr tracks its own lifecycle (creation, retention, release) and
    notifies the MemoryEngine for external memory management (e.g.,
    KV-cache eviction, hierarchical storage).
    
    Behaves exactly like str for all string operations.
    """
    
    def __new__(cls, content: str, *, 
                memory_engine: Optional["MemoryEngine"] = None,
                context: str = "unknown"):
        """Create a new MemStr.
        
        Args:
            content: The string content
            memory_engine: Optional engine to notify of lifecycle events
            context: Where this MemStr is created (for debugging/tracking)
        """
        if isinstance(content, MemStr):
            # If content is already MemStr, just use its string value
            content = str(content)
        
        obj = super().__new__(cls, content)
        obj._refcount = 0
        obj._memory_engine = memory_engine
        obj._context = context
        obj._retained = False
        
        # Notify engine of creation
        if obj._memory_engine:
            obj._memory_engine.on_memstr_create(obj, context)
        
        return obj
    
    def __add__(self, other):
        """Concatenation returns MemStr."""
        result = MemStr(
            super().__add__(str(other)),
            memory_engine=self._memory_engine,
            context="concat"
        )
        return result
    
    def __radd__(self, other):
        """Right concatenation returns MemStr."""
        result = MemStr(
            str(other) + str(self),
            memory_engine=self._memory_engine,
            context="concat"
        )
        return result
    
    def __getitem__(self, key):
        """Slicing returns MemStr."""
        result = super().__getitem__(key)
        if isinstance(result, str):
            return MemStr(result, memory_engine=self._memory_engine, context="slice")
        return result
    
    def replace(self, old, new, count=-1):
        """Replace returns MemStr."""
        result = super().replace(old, new, count)
        return MemStr(result, memory_engine=self._memory_engine, context="replace")
    
    def upper(self):
        """Uppercase returns MemStr."""
        return MemStr(super().upper(), memory_engine=self._memory_engine, context="upper")
    
    def lower(self):
        """Lowercase returns MemStr."""
        return MemStr(super().lower(), memory_engine=self._memory_engine, context="lower")
    
    def strip(self, chars=None):
        """Strip returns MemStr."""
        return MemStr(super().strip(chars), memory_engine=self._memory_engine, context="strip")
    
    def lstrip(self, chars=None):
        """Left strip returns MemStr."""
        return MemStr(super().lstrip(chars), memory_engine=self._memory_engine, context="lstrip")
    
    def rstrip(self, chars=None):
        """Right strip returns MemStr."""
        return MemStr(super().rstrip(chars), memory_engine=self._memory_engine, context="rstrip")
    
    def format(self, *args, **kwargs):
        """Format returns MemStr."""
        return MemStr(super().format(*args, **kwargs), 
                     memory_engine=self._memory_engine, context="format")
    
    def _retain(self):
        """Increment refcount - called when MemStr is stored (stack, variables)."""
        self._refcount += 1
        self._retained = True
    
    def _release(self):
        """Decrement refcount - called when MemStr is no longer referenced."""
        if self._refcount > 0:
            self._refcount -= 1
            if self._refcount == 0 and self._memory_engine:
                self._memory_engine.on_memstr_drop(self)
                self._retained = False
    
    def _set_engine(self, engine: Optional["MemoryEngine"]):
        """Set the memory engine (used when MemStr moves between contexts)."""
        self._memory_engine = engine
    
    def __repr__(self):
        return f"MemStr({super().__repr__()})"
    
    def __str__(self):
        # Return plain str to avoid MemStr proliferation in external APIs
        return super().__str__()


def to_memstr(obj, memory_engine: Optional["MemoryEngine"] = None, context: str = "convert") -> MemStr:
    """Convert any object to MemStr.
    
    If obj is already MemStr, returns it (optionally updating engine).
    Otherwise creates new MemStr from str(obj).
    """
    if isinstance(obj, MemStr):
        if memory_engine and obj._memory_engine is None:
            obj._set_engine(memory_engine)
        return obj
    return MemStr(str(obj), memory_engine=memory_engine, context=context)
