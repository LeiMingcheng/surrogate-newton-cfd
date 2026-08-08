"""Canonical model registry for Surrogate-Newton CFD surrogate models."""

from typing import TYPE_CHECKING, Any, Dict, Type, Union

if TYPE_CHECKING:
    from surrogate.common import BaseSurrogate


class ModelRegistry:
    """Registry for model classes"""

    _public_model_keys = ("direct_fno", "direct_dit", "fsb_fno", "fsb_dit")
    _models: Dict[str, Type["BaseSurrogate"]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type["BaseSurrogate"]):
        """Register a model class"""
        name = str(name).lower()
        if name not in cls._public_model_keys:
            raise ValueError(f"Cannot register non-public model key: {name!r}")
        cls._models[name.lower()] = model_class

    @classmethod
    def get(cls, name: str) -> Type["BaseSurrogate"]:
        """Get a model class by name"""
        if not cls._models:
            register_models()
        key = str(name).lower()
        if key not in cls._models:
            raise ValueError(
                f"Unknown model key: {name}. "
                f"Public model keys: {cls.list_models()}"
            )
        return cls._models[key]

    @classmethod
    def list_models(cls) -> list:
        """List canonical public model names."""
        return list(cls._public_model_keys)


_MODELS_REGISTERED = False


def register_models():
    """Register all available models"""
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return

    from surrogate.direct import DirectDiT, DirectFNO
    from surrogate.fsb import FSBDiT, FSBFNO

    ModelRegistry.register("direct_fno", DirectFNO)
    ModelRegistry.register("direct_dit", DirectDiT)
    ModelRegistry.register("fsb_fno", FSBFNO)
    ModelRegistry.register("fsb_dit", FSBDiT)

    _MODELS_REGISTERED = True


def create_model(config: Union[Dict[str, Any], str, 'ModelConfig']) -> "BaseSurrogate":
    """
    Create a model from configuration

    Args:
        config: ModelConfig object, public model key string, or clean dict

    Returns:
        Instantiated model
    """
    # Import here to avoid circular dependency
    from surrogate.configs import ModelConfig

    if isinstance(config, ModelConfig):
        model_key = config.get_public_model_key()
        model_params = config.get_model_params()
    elif isinstance(config, str):
        model_key = str(config).lower()
        model_params = {}
    elif isinstance(config, dict):
        model_config = ModelConfig(**config)
        model_key = model_config.get_public_model_key()
        model_params = model_config.get_model_params()
    else:
        raise TypeError(f"Invalid config type: {type(config)}")

    register_models()

    model_class = ModelRegistry.get(model_key)

    return model_class(**model_params)


__all__ = [
    "ModelRegistry",
    "create_model",
]
