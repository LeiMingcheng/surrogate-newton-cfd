"""Serving contracts and predictor adapters for surrogate runtimes."""

from surrogate.serving.aoa import AoASolverConfig, SurrogateAoASolver
from surrogate.serving.batching import BatchingConfig, DynamicBatcher
from surrogate.serving.client import SurrogateClient, SurrogateClientConfig
from surrogate.serving.contracts import (
    AoARequest,
    AoAResult,
    OnlineSample,
    PredictionRequest,
    PredictionResponse,
)
from surrogate.serving.online import (
    AsyncOnlineSampleWriter,
    OnlineSampleRecord,
    SQLiteOnlineBuffer,
    compute_geometry_id,
    compute_sample_id,
)
from surrogate.serving.predictors import DirectServingPredictor, FSBServingPredictor

__all__ = [
    "AoARequest",
    "AoAResult",
    "AoASolverConfig",
    "AsyncOnlineSampleWriter",
    "BatchingConfig",
    "DirectServingPredictor",
    "DynamicBatcher",
    "FSBServingPredictor",
    "OnlineSample",
    "OnlineSampleRecord",
    "PredictionRequest",
    "PredictionResponse",
    "SQLiteOnlineBuffer",
    "SurrogateAoASolver",
    "SurrogateClient",
    "SurrogateClientConfig",
    "compute_geometry_id",
    "compute_sample_id",
]
