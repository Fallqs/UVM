"""LM - Language Model wrapper for UVM.

Provides OpenAI API-based LLM calling with automatic MemStr conversion.
"""

from typing import Any, Dict, Optional, Union, TYPE_CHECKING
from pathlib import Path

from .config import load_config, merge_with_defaults
from .memstr import MemStr, to_memstr

if TYPE_CHECKING:
    from .context import UVMContext


class LM:
    """Language Model wrapper.
    
    Automatically converts inputs/outputs to MemStr and integrates with
    the memory engine for lifecycle tracking.
    
    Can be called directly or referenced by name in CALL instructions.
    """
    
    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        name: Optional[str] = None,
        memory_engine: Optional[Any] = None,
        context: Optional["UVMContext"] = None
    ):
        """Initialize LM.
        
        Args:
            config: Configuration dict, JSON path, or YAML path
            name: Optional name for registration in context
            memory_engine: MemoryEngine for MemStr tracking
            context: UVMContext for auto-registration
        """
        self.config = merge_with_defaults(load_config(config))
        self.name = name
        self._memory_engine = memory_engine
        
        # Auto-get context if not provided
        if context is None and name is not None:
            try:
                # Import from parent module to avoid circular import
                import uvm
                context = uvm._ensure_context()
            except Exception:
                pass  # No context available
        
        # Initialize OpenAI client (if API key available)
        self.client = None
        api_key = self.config.get("api_key")
        if api_key:
            try:
                from openai import OpenAI
                
                # Extract OpenAI-specific config
                client_kwargs = {"api_key": api_key}
                if "base_url" in self.config:
                    client_kwargs["base_url"] = self.config["base_url"]
                
                self.client = OpenAI(**client_kwargs)
            except ImportError:
                pass  # OpenAI not installed, use mock mode
        
        # Register if name provided
        if name and context:
            context.register_lm(name, self)
    
    def __call__(self, *args) -> MemStr:
        """Call the LM with given arguments.
        
        All args are converted to MemStr, joined with newlines,
        and sent to the LLM. Returns a MemStr with the response.
        
        Args:
            *args: Variable arguments (strings, MemStrs, or anything str-able)
        
        Returns:
            MemStr containing the LLM response.
        """
        # Convert all args to MemStr and join
        parts = []
        for arg in args:
            ms = to_memstr(arg, memory_engine=self._memory_engine, context="lm_input")
            parts.append(ms)
        
        prompt = "\n".join(str(p) for p in parts)
        
        # Mock mode: if no client available, return mock response
        if self.client is None:
            mock_response = f"[MOCK LM RESPONSE] Received: {prompt[:100]}..."
            return MemStr(mock_response, memory_engine=self._memory_engine, context="lm_output")
        
        # Prepare API call
        model = self.config.get("model", "gpt-4o-mini")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1024)
        top_p = self.config.get("top_p", 1.0)
        frequency_penalty = self.config.get("frequency_penalty", 0.0)
        presence_penalty = self.config.get("presence_penalty", 0.0)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if present in config
        if "system" in self.config:
            messages.insert(0, {"role": "system", "content": self.config["system"]})
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        
        # Extract result
        content = response.choices[0].message.content
        
        # Return as MemStr
        return MemStr(content, memory_engine=self._memory_engine, context="lm_output")
    
    def __repr__(self):
        return f"LM(name={self.name!r}, model={self.config.get('model')!r})"
