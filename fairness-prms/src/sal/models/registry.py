from typing import Dict, Type
from sal.models.prm import (
    PRM,
    BiasDetectionPRM,
)

PRM_REGISTRY: Dict[str, Type[PRM]] = {
    "zarahall/bias-prm-v3": BiasDetectionPRM,
}

def register_prm(model_id: str, prm_class: Type[PRM]):
    """Register a new PRM class."""
    PRM_REGISTRY[model_id] = prm_class

def get_prm_class(model_id: str) -> Type[PRM]:
    """Get PRM class for a model ID."""
    if model_id not in PRM_REGISTRY:
        raise ValueError(f"No PRM class found for model {model_id}")
    return PRM_REGISTRY[model_id] 