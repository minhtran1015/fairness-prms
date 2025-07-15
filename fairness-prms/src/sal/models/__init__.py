from .prm import PRM
from .registry import get_prm_class

def load_prm(config):
    """Load multiple PRM models based on the configuration."""
    prms = []
    for prm_path, prm_weight in zip(config.model.prm_paths, config.model.prm_weights):
        prm_class = get_prm_class(prm_path)
        prm = prm_class(config)
        print(f"Loaded PRM from {prm_path} with weight {prm_weight}")
        prm.weight = prm_weight  # Add weight to the PRM instance
        prms.append(prm)
    return prms

__all__ = ["PRM", "load_prm"]
