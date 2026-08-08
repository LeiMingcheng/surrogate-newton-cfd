"""Physics utilities shared by surrogate training, inference, and evaluation."""

from surrogate.physics.losses import compute_volume_weighted_mse

__all__ = ["compute_volume_weighted_mse"]
