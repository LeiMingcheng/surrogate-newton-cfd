"""Flow-perceptual encoder modules used by clean FSB training losses."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogate.common.components.conditioning import FiLM


class OGridConv2d(nn.Module):
    """Conv2d with O-grid aware padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.pad = int(kernel_size) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="reflect")
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and O-grid convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = OGridConv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = OGridConv2d(out_channels, out_channels, kernel_size=3, stride=1)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.norm1(x))
        y = self.conv1(y)
        y = self.act2(self.norm2(y))
        y = self.conv2(y)
        return y + self.skip(x)


class FlowPerceptualEncoder(nn.Module):
    """Four-stage ResNet-style encoder with FiLM flow-condition conditioning."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        channels = list(config.get("channels", [64, 128, 256, 512]))
        input_channels = int(config.get("input_channels", 15))
        condition_dim = int(config.get("condition_dim", 3))
        num_groups = int(config.get("num_groups", 8))

        self.stem = nn.Sequential(
            OGridConv2d(input_channels, channels[0], kernel_size=7, stride=2),
            nn.GroupNorm(min(num_groups, channels[0]), channels[0]),
            nn.SiLU(),
        )
        self.film_stem = FiLM(condition_dim, channels[0], activation="silu")
        self.stage2 = ResBlock(channels[0], channels[1], stride=2, num_groups=num_groups)
        self.film2 = FiLM(condition_dim, channels[1], activation="silu")
        self.stage3 = ResBlock(channels[1], channels[2], stride=2, num_groups=num_groups)
        self.film3 = FiLM(condition_dim, channels[2], activation="silu")
        self.stage4 = ResBlock(channels[2], channels[3], stride=2, num_groups=num_groups)
        self.film4 = FiLM(condition_dim, channels[3], activation="silu")

    @staticmethod
    def _preprocess_flow_conditions(flow_conditions: torch.Tensor) -> torch.Tensor:
        if flow_conditions.shape[1] < 3:
            return flow_conditions
        mach = flow_conditions[:, 0:1]
        aoa_rad = torch.deg2rad(flow_conditions[:, 1:2])
        reynolds = flow_conditions[:, 2:3].clamp(min=1.0e3)
        if (flow_conditions[:, 2] < 10.0).all():
            return flow_conditions
        return torch.cat([mach, aoa_rad, torch.log10(reynolds)], dim=1)

    def forward(self, x: torch.Tensor, flow_conditions: torch.Tensor) -> List[torch.Tensor]:
        cond = self._preprocess_flow_conditions(flow_conditions)
        phi1 = self.film_stem(self.stem(x), cond)
        phi2 = self.film2(self.stage2(phi1), cond)
        phi3 = self.film3(self.stage3(phi2), cond)
        phi4 = self.film4(self.stage4(phi3), cond)
        return [phi1, phi2, phi3, phi4]

    def extract_features(
        self,
        x: torch.Tensor,
        flow_conditions: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if flow_conditions is None:
            flow_conditions = torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)
            flow_conditions[:, 0] = 0.5
            flow_conditions[:, 2] = 1.0e6
        return self.forward(x, flow_conditions)


def _match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    if upsampled.shape[2:] != skip.shape[2:]:
        upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False)
    return torch.cat([upsampled, skip], dim=1)


class FlowPerceptualDecoder(nn.Module):
    """U-Net-style decoder paired with FlowPerceptualEncoder."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        channels = list(config.get("channels", [64, 128, 256, 512]))
        output_channels = int(config.get("input_channels", 15))
        num_groups = int(config.get("num_groups", 8))
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = ResBlock(channels[3] + channels[2], channels[2], num_groups=num_groups)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ResBlock(channels[2] + channels[1], channels[1], num_groups=num_groups)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResBlock(channels[1] + channels[0], channels[0], num_groups=num_groups)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = OGridConv2d(channels[0], output_channels, kernel_size=3)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = features
        x = self.up4(phi4)
        x = self.dec4(_match_and_cat(x, phi3))
        x = self.up3(x)
        x = self.dec3(_match_and_cat(x, phi2))
        x = self.up2(x)
        x = self.dec2(_match_and_cat(x, phi1))
        x = self.up1(x)
        return self.final_conv(x)


__all__ = ["FlowPerceptualDecoder", "FlowPerceptualEncoder"]
