import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
from sal.config.base import Config, ModelConfig, DatasetConfig, OutputConfig, SearchConfig

def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_recipe(recipe_name: str) -> Dict[str, Any]:
    """Load a predefined configuration recipe."""
    recipe_path = Path(__file__).parent.parent.parent.parent / "recipes" /  f"{recipe_name}.yaml"
    if not recipe_path.exists():
        raise ValueError(f"Recipe {recipe_name} not found at {recipe_path}")
    
    with open(recipe_path) as f:
        return yaml.safe_load(f)

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    recipe: Optional[str] = None,
    **overrides
) -> Config:
    """Load configuration from multiple sources with priority:
    1. Base config from recipe (if provided)
    2. Config file (if provided)
    3. Override parameters
    
    Args:
        config_path: Path to a YAML configuration file
        recipe: Name of a predefined recipe to use as base config
        **overrides: Keyword arguments to override specific config values
    """
    config_dict = {}
    
    # 1. Load recipe if provided
    if recipe:
        config_dict = load_recipe(recipe)
    
    # 2. Load and merge config file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
            config_dict = _deep_update(config_dict, file_config)
    
    # 3. Apply overrides
    # Convert flat overrides to nested dict structure
    override_dict = {}
    for key, value in overrides.items():
        parts = key.split('.')
        current = override_dict
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    
    config_dict = _deep_update(config_dict, override_dict)
    
    # Create config objects
    model_config = ModelConfig(**config_dict.get("model", {}))
    dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
    output_config = OutputConfig(**config_dict.get("output", {}))
    search_config = SearchConfig(**config_dict.get("search", {}))
    
    return Config(
        model=model_config,
        dataset=dataset_config,
        output=output_config,
        search=search_config
    ) 