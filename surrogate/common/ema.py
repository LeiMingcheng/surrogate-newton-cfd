"""
Exponential Moving Average (EMA) model utilities

Provides EMA parameter tracking for stable model evaluation during training.
"""

import torch
import torch.nn as nn


def _strip_module_prefix(key: str) -> str:
    """Strip 'module.' prefix from DDP parameter names for consistent checkpointing."""
    if key.startswith('module.'):
        return key[7:]  # len('module.') = 7
    return key


class EMAModel:
    """
    Exponential Moving Average model for stable evaluation

    Maintains a shadow copy of model parameters with EMA updates.

    Note on DDP compatibility:
    - Internal shadow dict uses raw parameter names (may have 'module.' prefix in DDP)
    - state_dict() normalizes keys by stripping 'module.' prefix for checkpoint portability
    - load_state_dict() handles both formats for cross-mode compatibility
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: The model to track
            decay: EMA decay factor (higher = slower update)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(
                    param.data,
                    alpha=1.0 - self.decay,
                )

    def apply_shadow(self):
        """Apply shadow parameters to model without allocating a second backup."""
        if self.backup:
            raise RuntimeError("EMA shadow parameters are already applied")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """
        Return state dict for checkpointing.

        Keys are normalized (module. prefix stripped) for cross-mode compatibility.
        This ensures checkpoints saved in DDP mode can be loaded in single-GPU mode and vice versa.
        """
        # Normalize shadow keys: strip 'module.' prefix for checkpoint portability
        normalized_shadow = {
            _strip_module_prefix(k): v for k, v in self.shadow.items()
        }
        return {
            'shadow': normalized_shadow,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """
        Load state dict from checkpoint.

        Handles cross-mode compatibility (DDP <-> single-GPU):
        - First tries exact key match
        - Then tries with 'module.' prefix stripped/added
        - New parameters (not in checkpoint) are initialized from current model
        """
        loaded_shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)

        # Build lookup dict with normalized keys for flexible matching
        # This handles: DDP->DDP, DDP->single, single->DDP, single->single
        normalized_lookup = {}
        for k, v in loaded_shadow.items():
            normalized_key = _strip_module_prefix(k)
            normalized_lookup[normalized_key] = v

        missing_keys = []
        loaded_keys = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Try matching with normalized key (handles all DDP/single-GPU combinations)
                normalized_name = _strip_module_prefix(name)

                if normalized_name in normalized_lookup:
                    self.shadow[name] = normalized_lookup[normalized_name]
                    loaded_keys.append(name)
                else:
                    # New parameter not in checkpoint: initialize from current value
                    self.shadow[name] = param.data.clone()
                    missing_keys.append(name)

        if missing_keys:
            print(f"  EMA: Initialized {len(missing_keys)} new params from model: {missing_keys}")
