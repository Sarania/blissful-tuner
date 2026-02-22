#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility training functions adapted from kohya-ss/sdscripts

License: Apache-2.0
Created on Thu Apr 24 11:29:37 2025
Author: Blyss
"""

import torch
import random
import torch.nn.functional as F
from typing import Optional


# originally from https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(
    noise: torch.Tensor,
    device: Optional[torch.device] = None,
    iterations: int = 6,
    discount: float = 0.4,
    channels_last: bool = False,
) -> torch.Tensor:
    """
    Adds multi-scale (pyramid) noise and rescales to ~unit variance.
    Supports:
      - 4D: BCHW (channels_last=False) or BHWC (channels_last=True)
      - 5D: BCFHW (channels_last=False) or BFHWC (channels_last=True)
    For 5D, frames are treated as extra batch items (B*F) for bilinear scaling.
    """
    if device is None:
        device = noise.device

    dims = noise.dim()
    if dims not in (4, 5):
        raise ValueError(f"Expected 4D or 5D noise, got {dims}D with shape {tuple(noise.shape)}")

    # Move to device (and keep dtype consistent)
    noise = noise.to(device)

    # Convert to NCHW (4D) for interpolate.
    # For 5D, fold frames into batch: (B, C, F, H, W) -> (B*F, C, H, W)
    # For channels_last, permute first.
    if dims == 4:
        if channels_last:
            # BHWC -> BCHW
            x = noise.permute(0, 3, 1, 2).contiguous()
        else:
            x = noise
        b, c, h, w = x.shape
        bf = b  # effective batch

    else:  # dims == 5
        if channels_last:
            # BFHWC -> BCFHW
            x = noise.permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = noise
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * f, c, h, w)  # (B*F, C, H, W)
        bf = b * f  # effective batch

    # Base size is (H, W) (note order!)
    base_h, base_w = h, w

    for i in range(iterations):
        r = random.random() * 2 + 2  # in [2, 4)
        hn = max(1, int(base_h / (r**i)))
        wn = max(1, int(base_w / (r**i)))

        # Low-res noise then upsample to (base_h, base_w)
        low = torch.randn((bf, c, hn, wn), device=device, dtype=x.dtype)
        up = F.interpolate(low, size=(base_h, base_w), mode="bilinear", align_corners=False)

        x = x + up * (discount**i)

        if hn == 1 or wn == 1:
            break

    # Normalize variance back to ~1 (match original intent)
    x = x / x.std()

    # Convert back to the original layout
    if dims == 4:
        if channels_last:
            # BCHW -> BHWC
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    else:
        # (B*F, C, H, W) -> (B, F, C, H, W) -> BCFHW
        x = x.view(b, f, c, base_h, base_w).permute(0, 2, 1, 3, 4).contiguous()

        if channels_last:
            # BCFHW -> BFHWC
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x


# originally from https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_offset: Optional[torch.Tensor],
    adaptive_noise_scale: Optional[torch.Tensor],
    channels_last: bool = False,
    include_frames_in_mean: bool = True,  # for 5D latents: mean over F too (closer to 4D semantics)
) -> torch.Tensor:
    if noise_offset is None:
        return noise

    dims = latents.dim()
    if dims not in (4, 5):
        raise ValueError(f"Expected latents to be 4D or 5D, got {dims}D with shape {tuple(latents.shape)}")

    # Figure out which axis is channels, and which axes are "spatial" (and optionally frames)
    if dims == 4:
        # BCHW or BHWC
        cdim = 1 if not channels_last else 3
        reduce_dims = (2, 3) if not channels_last else (1, 2)  # H,W
    else:
        # BCFHW or BFHWC
        cdim = 1 if not channels_last else 4
        if not channels_last:
            # (B, C, F, H, W)
            reduce_dims = (2, 3, 4) if include_frames_in_mean else (3, 4)
        else:
            # (B, F, H, W, C)
            reduce_dims = (1, 2, 3) if include_frames_in_mean else (2, 3)

    if adaptive_noise_scale is not None:
        latent_mean = latents.mean(dim=reduce_dims, keepdim=True).abs()
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)

    # Build a randn shape that is 1 everywhere except batch + channels (+ frames if per-frame offsets)
    rand_shape = [1] * dims
    rand_shape[0] = latents.shape[0]  # batch
    rand_shape[cdim] = latents.shape[cdim]  # channels

    if dims == 5 and not include_frames_in_mean:
        # If we're doing per-frame offsets, keep the frame dimension too
        fdim = 2 if not channels_last else 1
        rand_shape[fdim] = latents.shape[fdim]

    noise = noise + noise_offset * torch.randn(tuple(rand_shape), device=latents.device, dtype=noise.dtype)
    return noise
