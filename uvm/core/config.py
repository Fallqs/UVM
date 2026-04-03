"""Configuration loader for UVM.

Supports dict, JSON, and YAML configuration formats.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def load_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load configuration from various sources.
    
    Args:
        config: Either a dict, path to JSON file, or path to YAML file.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        ValueError: If config type is unsupported or file not found.
    """
    if isinstance(config, dict):
        return _expand_env_vars(config)
    
    config_path = Path(config)
    
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config}")
    
    content = config_path.read_text(encoding='utf-8')
    suffix = config_path.suffix.lower()
    
    if suffix in ['.json']:
        cfg = json.loads(content)
    elif suffix in ['.yml', '.yaml']:
        try:
            import yaml
            cfg = yaml.safe_load(content)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
    else:
        # Try JSON first, then YAML
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml
                cfg = yaml.safe_load(content)
            except ImportError:
                raise ValueError(
                    f"Cannot parse config file {config}. "
                    "Supported formats: .json, .yml, .yaml"
                )
    
    return _expand_env_vars(cfg)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config values.
    
    Supports ${VAR} and ${VAR:-default} syntax.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return _expand_env_string(obj)
    else:
        return obj


def _expand_env_string(s: str) -> str:
    """Expand environment variables in a string."""
    import re
    
    # Pattern: ${VAR} or ${VAR:-default}
    pattern = r'\$\{([^}:-]+)(?::-([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default_val = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default_val is not None:
            return default_val
        raise ValueError(f"Environment variable {var_name} not set")
    
    return re.sub(pattern, replacer, s)


# Default configuration schema for LM
default_lm_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


def merge_with_defaults(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user config with defaults."""
    result = default_lm_config.copy()
    result.update(user_config)
    return result
