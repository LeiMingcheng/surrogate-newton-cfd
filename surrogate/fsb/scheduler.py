"""
I2SB (Image-to-Image Schrodinger Bridge) Scheduler

Implements Schrodinger Bridge scheduling for the FSB x1->x0 mapping.

Reference: I2SB: Image-to-Image Schrodinger Bridge (arXiv:2302.05872)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any


class I2SBScheduler(nn.Module):
    """
    I2SB Scheduler

    Bridge state:
    - x_t = w_0(t)*x_0 + w_1(t)*x_1 + sqrt(Sigma_t)*z

    Where:
    - x_0: Ground Truth (target)
    - x_1: FSB initial state
    - w_0, w_1: Time-dependent weight coefficients
    - Sigma_t: Time-dependent variance
    """

    def __init__(self,
                 num_timesteps: int = 1000,
                 beta_max: float = 0.3,
                 beta_schedule: str = 'symmetric_sine',
                 timestep_spacing: str = 'quadratic',
                 clip_sample: bool = False,
                 clip_sample_range: Tuple[float, float] = (-5.0, 5.0),
                 prediction_type: str = 'epsilon'):
        """
        Args:
            num_timesteps: Number of discrete timesteps (default 1000)
            beta_max: Maximum value of beta schedule (default 0.3)
            beta_schedule: Beta schedule type
                - 'symmetric_sine': beta_t = beta_max*sin(pi*t)^2 (recommended)
                - 'linear': beta_t = beta_max (constant)
                - 'cosine': cosine schedule
            timestep_spacing: Timestep distribution
                - 'uniform': Uniform distribution
                - 'quadratic': Quadratic distribution (dense at endpoints)
            clip_sample: Whether to clip samples
            clip_sample_range: Clipping range
            prediction_type: Network prediction type
                - 'epsilon': predict (x_t - x_0) / sigma_t (default)
                - 'v_prediction': predict v = epsilon - sigma_t * x_0
                - 'x0': directly predict x_0 (most direct supervision)
        """
        super().__init__()

        self.num_timesteps = num_timesteps
        self.beta_max = beta_max
        self.beta_schedule = beta_schedule
        self.timestep_spacing = timestep_spacing
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type

        if prediction_type not in ('epsilon', 'v_prediction', 'x0'):
            raise ValueError(f"Unknown prediction_type: {prediction_type}. Must be 'epsilon', 'v_prediction', or 'x0'")

        # Build discretized beta sequence
        betas = self._build_beta_schedule()
        self.register_buffer('betas', betas)

        # Precompute cumulative variances sigma_t^2 and sigma_bar_t^2
        # FIX: Discrete integral must multiply by dt to match continuous formulation
        # sigma_t^2 = ∫_0^t β(s) ds ≈ Σ_{k≤t} β_k · Δt
        dt = 1.0 / self.num_timesteps
        sigma_sq = torch.cumsum(betas * dt, dim=0)
        total = sigma_sq[-1]  # = ∫_0^1 β(t) dt ≈ 0.5*beta_max for symmetric_sine
        sigma_bar_sq = total - sigma_sq  # sigma_bar_t^2 = total - sigma_t^2

        self.register_buffer('sigma_sq', sigma_sq)           # (T,)
        self.register_buffer('sigma_bar_sq', sigma_bar_sq)   # (T,)
        self.register_buffer('total_variance', total.detach().clone())  # scalar

        # Precompute weight coefficients (w0 + w1 = 1)
        w0 = sigma_bar_sq / (total + 1e-8)  # x_0 (GT) weight
        w1 = sigma_sq / (total + 1e-8)       # x_1 (Surrogate) weight

        self.register_buffer('w0', w0)  # (T,)
        self.register_buffer('w1', w1)  # (T,)

        # Precompute mixture variance Sigma_t = (σ_t² · σ̄_t²) / total
        variance = (sigma_sq * sigma_bar_sq) / (total + 1e-8)
        self.register_buffer('variance', variance)              # (T,)
        self.register_buffer('std', torch.sqrt(variance + 1e-8))  # (T,)

        # Precompute sigma_t (for training target normalization)
        self.register_buffer('sigma', torch.sqrt(sigma_sq + 1e-8))  # (T,)
        # Precompute 1 + sigma_t^2 (for v-prediction x0 reconstruction)
        self.register_buffer('one_plus_sigma_sq', 1.0 + sigma_sq)  # (T,)

    def _build_beta_schedule(self) -> torch.Tensor:
        """Build beta schedule sequence"""
        t = torch.linspace(0, 1, self.num_timesteps)

        if self.beta_schedule == 'symmetric_sine':
            # Symmetric sine schedule: beta_t = beta_max * sin(pi*t)^2
            # Property: beta=0 at t=0 and t=1, beta=beta_max at t=0.5
            betas = self.beta_max * torch.sin(math.pi * t) ** 2
        elif self.beta_schedule == 'linear':
            # Constant schedule
            betas = torch.full((self.num_timesteps,), self.beta_max)
        elif self.beta_schedule == 'cosine':
            # Cosine schedule.
            s = 0.008
            steps = self.num_timesteps
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        return betas.float()

    def get_timesteps(self,
                      num_inference_steps: int,
                      device: torch.device) -> torch.Tensor:
        """
        Get inference timestep sequence

        Args:
            num_inference_steps: Number of inference steps
            device: Device

        Returns:
            timesteps: (num_inference_steps,) timestep sequence from high to low
        """
        if self.timestep_spacing == 'uniform':
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, num_inference_steps,
                device=device
            ).long()
        elif self.timestep_spacing == 'quadratic':
            # Quadratic distribution: dense sampling at endpoints
            u = torch.linspace(0, 1, num_inference_steps, device=device)
            # Two-end dense transformation
            t_normalized = torch.where(
                u < 0.5,
                2 * u ** 2,           # [0, 0.5] -> [0, 0.5]
                1 - 2 * (1 - u) ** 2  # [0.5, 1] -> [0.5, 1]
            )
            # Reverse (from high to low) and map to [0, T-1]
            timesteps = ((1 - t_normalized) * (self.num_timesteps - 1)).long()
        else:
            raise ValueError(f"Unknown timestep_spacing: {self.timestep_spacing}")

        return timesteps

    def add_bridge_noise(self,
                         x0: torch.Tensor,
                         x1: torch.Tensor,
                         timesteps: torch.Tensor,
                         noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        I2SB forward process: generate intermediate state x_t from x_0 and x_1

        Formula: x_t = w_0(t)*x_0 + w_1(t)*x_1 + sqrt(Sigma_t)*z

        Args:
            x0: (B, C, H, W) Ground Truth
            x1: (B, C, H, W) Surrogate prediction
            timesteps: (B,) timesteps
            noise: (B, C, H, W) optional noise, randomly generated if None

        Returns:
            x_t: (B, C, H, W) intermediate state
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Get coefficients for current timesteps (B,) -> (B, 1, 1, 1)
        w0_t = self.w0[timesteps].view(-1, 1, 1, 1)
        w1_t = self.w1[timesteps].view(-1, 1, 1, 1)
        std_t = self.std[timesteps].view(-1, 1, 1, 1)

        # I2SB forward sampling
        x_t = w0_t * x0 + w1_t * x1 + std_t * noise

        return x_t

    def get_training_target(self,
                            x_t: torch.Tensor,
                            x0: torch.Tensor,
                            timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get training target (network prediction direction)

        For epsilon prediction:
            target = (x_t - x_0) / sigma_t

        For v-prediction:
            epsilon = (x_t - x_0) / sigma_t
            v = epsilon - sigma_t * x_0

        For x0 prediction:
            target = x_0 (direct supervision)

        Args:
            x_t: (B, C, H, W) current state
            x0: (B, C, H, W) Ground Truth
            timesteps: (B,) timesteps

        Returns:
            target: (B, C, H, W) training target
        """
        if self.prediction_type == 'x0':
            # Direct x0 prediction: target is simply x0
            return x0

        sigma_t = self.sigma[timesteps].view(-1, 1, 1, 1)
        epsilon = (x_t - x0) / (sigma_t + 1e-8)

        if self.prediction_type == 'epsilon':
            return epsilon
        elif self.prediction_type == 'v_prediction':
            # v = epsilon - sigma_t * x0
            v = epsilon - sigma_t * x0
            return v
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def compute_v(self,
                  x_t: torch.Tensor,
                  x0: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 to v-prediction space.

        v = epsilon - sigma_t * x0
          = (x_t - x0) / sigma_t - sigma_t * x0

        Used when prediction_type='x0' but loss is computed in v-space.
        """
        sigma_t = self.sigma[timesteps].view(-1, 1, 1, 1)
        epsilon = (x_t - x0) / (sigma_t + 1e-8)
        v = epsilon - sigma_t * x0
        return v

    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             x1: torch.Tensor,
             timestep_next: int,
             eta: float = 0.0,
             return_dict: bool = False) -> Dict[str, Any]:
        """
        I2SB reverse denoising step

        Reconstruct x_0 from x_t, then compute x_{t-1}

        Args:
            model_output: (B, C, H, W) network output (predicted direction)
            timestep: current timestep t
            sample: (B, C, H, W) current sample x_t
            x1: (B, C, H, W) Surrogate prediction (I2SB-specific parameter)
            timestep_next: next timestep t-1
            eta: randomness control parameter (0=deterministic, >0=stochastic)
            return_dict: whether to return detailed dictionary

        Returns:
            If return_dict=False: prev_sample (Tensor)
            If return_dict=True: dict with 'prev_sample', 'pred_original_sample', 'mean', 'sigma', 'z'
        """
        # Step 1: Reconstruct x_0 from network output
        sigma_t = self.sigma[timestep]
        if self.prediction_type == 'epsilon':
            # epsilon prediction: x_0 = x_t - sigma_t * epsilon
            pred_x0 = sample - sigma_t * model_output
        elif self.prediction_type == 'v_prediction':
            # v-prediction: x_0 = (x_t - sigma_t * v) / (1 + sigma_t^2)
            one_plus_sigma_sq = self.one_plus_sigma_sq[timestep]
            pred_x0 = (sample - sigma_t * model_output) / (one_plus_sigma_sq + 1e-8)
        elif self.prediction_type == 'x0':
            # x0 prediction: model output is directly x_0
            pred_x0 = model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Optional clipping
        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, *self.clip_sample_range)

        # Step 2: Compute x_{t-1} distribution parameters
        if timestep_next <= 0:
            # Final step: directly return pred_x0
            prev_sample = pred_x0
            mean = pred_x0
            sigma_step = torch.zeros_like(pred_x0)
            z = torch.zeros_like(pred_x0)
        else:
            # Intermediate step: use I2SB posterior distribution
            w0_next = self.w0[timestep_next]
            w1_next = self.w1[timestep_next]
            std_next = self.std[timestep_next]

            # Posterior mean: mu_{t-1} = w_0(t-1)*x_hat_0 + w_1(t-1)*x_1
            mean = w0_next * pred_x0 + w1_next * x1

            # Posterior standard deviation
            sigma_step = std_next * eta  # eta controls randomness

            # Sampling
            if eta > 0:
                z = torch.randn_like(mean)
                prev_sample = mean + sigma_step * z
            else:
                z = torch.zeros_like(mean)
                prev_sample = mean

        if return_dict:
            return {
                'prev_sample': prev_sample,
                'pred_original_sample': pred_x0,
                'mean': mean,
                'sigma': sigma_step if isinstance(sigma_step, torch.Tensor) else torch.full_like(mean, sigma_step),
                'z': z
            }
        else:
            return prev_sample

    def reconstruct_x0(self,
                       x_t: torch.Tensor,
                       model_output: torch.Tensor,
                       timesteps: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct x0 from x_t and model output (for external use)

        For epsilon prediction:
            x_0 = x_t - sigma_t * epsilon

        For v-prediction:
            x_0 = (x_t - sigma_t * v) / (1 + sigma_t^2)

        For x0 prediction:
            x_0 = model_output (direct)

        Args:
            x_t: (B, C, H, W) current state
            model_output: (B, C, H, W) network output
            timesteps: (B,) timesteps

        Returns:
            pred_x0: (B, C, H, W) reconstructed x0
        """
        if self.prediction_type == 'x0':
            # x0 prediction: model output is directly x_0
            return model_output

        sigma_t = self.sigma[timesteps].view(-1, 1, 1, 1)

        if self.prediction_type == 'epsilon':
            pred_x0 = x_t - sigma_t * model_output
        elif self.prediction_type == 'v_prediction':
            one_plus_sigma_sq = self.one_plus_sigma_sq[timesteps].view(-1, 1, 1, 1)
            pred_x0 = (x_t - sigma_t * model_output) / (one_plus_sigma_sq + 1e-8)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return pred_x0

    def step_from_x0(self,
                     timestep: int,
                     x0: torch.Tensor,
                     x1: torch.Tensor,
                     sample: torch.Tensor,
                     timestep_next: int,
                     eta: float = 0.0,
                     return_dict: bool = False) -> Dict[str, Any]:
        """
        Compute next step directly from given x0 (for use after Inpainting mask blending)

        Args:
            timestep: current timestep t
            x0: (B, C, H, W) blended x0 (e.g., mask-blended pred_x0)
            x1: (B, C, H, W) Surrogate prediction
            sample: (B, C, H, W) current sample x_t (for logging only, not used in computation)
            timestep_next: next timestep t-1
            eta: randomness control parameter
            return_dict: whether to return detailed dictionary

        Returns:
            Same as step()
        """
        if timestep_next <= 0:
            prev_sample = x0
            mean = x0
            sigma_step = torch.zeros_like(x0)
            z = torch.zeros_like(x0)
        else:
            w0_next = self.w0[timestep_next]
            w1_next = self.w1[timestep_next]
            std_next = self.std[timestep_next]

            mean = w0_next * x0 + w1_next * x1
            sigma_step = std_next * eta

            if eta > 0:
                z = torch.randn_like(mean)
                prev_sample = mean + sigma_step * z
            else:
                z = torch.zeros_like(mean)
                prev_sample = mean

        if return_dict:
            return {
                'prev_sample': prev_sample,
                'pred_original_sample': x0,
                'mean': mean,
                'sigma': sigma_step if isinstance(sigma_step, torch.Tensor) else torch.full_like(mean, sigma_step),
                'z': z
            }
        else:
            return prev_sample

    def get_timesteps_from_t0(self,
                              t0: int,
                              num_inference_steps: int,
                              device: torch.device) -> torch.Tensor:
        """
        Generate timestep sequence with quadratic distribution in [t0, 0] interval

        Unlike get_timesteps() which uses global [T-1, 0] range,
        this method generates timesteps within [t0, 0] with endpoint-dense sampling.

        Args:
            t0: Starting timestep (maximum value)
            num_inference_steps: Number of inference steps
            device: Device

        Returns:
            timesteps: (num_inference_steps,) timestep sequence from t0 to 0
        """
        u = torch.linspace(0, 1, num_inference_steps, device=device)

        if self.timestep_spacing == 'quadratic':
            # Two-end dense transformation: dense at u=0 (t0) and u=1 (0)
            t_normalized = torch.where(
                u < 0.5,
                2 * u ** 2,           # [0, 0.5] -> [0, 0.5] quadratic
                1 - 2 * (1 - u) ** 2  # [0.5, 1] -> [0.5, 1] quadratic
            )
        else:
            # uniform: linear distribution
            t_normalized = u

        # Map to [t0, 0], from high to low
        # t_normalized=0 -> t0, t_normalized=1 -> 0
        timesteps = (t0 * (1 - t_normalized)).long()

        return timesteps

    def add_noise_from_x1(self,
                          x1: torch.Tensor,
                          timesteps: torch.Tensor,
                          noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x1-only bridge sampling: add I2SB variance noise from the initializer state

        Formula: x_t = x_1 + std_t * z

        This is a special case of add_bridge_noise with w0=0, w1=1.
        Used when a target state is not available.

        Args:
            x1: (B, C, H, W) Surrogate prediction
            timesteps: (B,) timesteps
            noise: (B, C, H, W) optional noise, randomly generated if None

        Returns:
            x_t: (B, C, H, W) noisy state
        """
        if noise is None:
            noise = torch.randn_like(x1)

        std_t = self.std[timesteps].view(-1, 1, 1, 1)
        x_t = x1 + std_t * noise

        return x_t
