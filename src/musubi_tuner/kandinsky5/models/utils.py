# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import math

import torch

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def exist(item):
    return item is not None


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)
    return freqs


def fractal_flatten(x, rope, shape, block_mask=False):
    if block_mask:
        pixel_size = 8
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=0)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=0)
        x = x.flatten(0, 1)
        rope = rope.flatten(0, 1)
    else:
        x = x.flatten(0, 2)
        rope = rope.flatten(0, 2)
    return x, rope


def fractal_unflatten(x, shape, block_mask=False):
    if block_mask:
        pixel_size = 8
        x = x.reshape(-1, pixel_size**2, *x.shape[1:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=0)
    else:
        x = x.reshape(*shape, *x.shape[1:])
    return x


def local_patching(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration // g1, g1, height // g2, g2, width // g3, g3, *x.shape[dim + 3 :])
    x = x.permute(*range(len(x.shape[:dim])), dim, dim + 2, dim + 4, dim + 1, dim + 3, dim + 5, *range(dim + 6, len(x.shape)))
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration // g1, height // g2, width // g3, g1, g2, g3, *x.shape[dim + 2 :])
    x = x.permute(*range(len(x.shape[:dim])), dim, dim + 3, dim + 1, dim + 4, dim + 2, dim + 5, *range(dim + 6, len(x.shape)))
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def fast_sta_nabla(T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device="cuda") -> Tensor:
    l = torch.Tensor([T, H, W]).amax()
    r = torch.arange(0, l, 1, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = (
        mat[:T, :T].flatten(),
        mat[:H, :H].flatten(),
        mat[:W, :W].flatten(),
    )
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
    return sta.reshape(T * H * W, T * H * W)


def nablaT_v2(q, k, sta, thr=0.9, add_sta=True, method="topcdf", BLOCK=128):
    # q,k: [B, h, S, D]
    B, h, S, D = q.shape

    if S % BLOCK != 0:
        raise ValueError(f"NABLA requires S divisible by BLOCK. Got S={S}, BLOCK={BLOCK}")

    s1 = S // BLOCK  # number of query/key blocks

    # Downsample q/k into blocks of size BLOCK
    qa = q.reshape(B, h, s1, BLOCK, D).mean(-2)
    ka = k.reshape(B, h, s1, BLOCK, D).mean(-2).transpose(-2, -1)
    attn_map = qa @ ka  # [B, h, s1, s1]
    attn_map = torch.softmax(attn_map / math.sqrt(D), dim=-1)

    # Binarize map into mask [B,h,s1,s1]
    vals, inds = attn_map.sort(-1)
    if method == "topk":
        k_top = int(thr * vals.shape[-1]) if 0 < thr < 1 else int(thr)
        k_top = max(1, min(k_top, vals.shape[-1]))
        mask = torch.zeros_like(vals, dtype=torch.bool)
        topk_inds = inds[..., -k_top:]
        mask.scatter_(-1, topk_inds, True)
    else:
        cvals = vals.cumsum(-1)
        mask = cvals >= 1 - thr
        mask = mask.gather(-1, inds.argsort(-1))

    if add_sta and sta is not None:
        # sta should become [B,h,s1,s1] and bool

        sta_b = sta.to(torch.bool)

        # If sta is [B,1,*,*], expand to heads
        if sta_b.dim() == 4 and sta_b.shape[1] == 1:
            # expand heads later after fixing spatial dims
            pass

        # Fix spatial dims: allow sta built for 64-block grid to be used with 128-block grid
        if sta_b.shape[-1] != s1 or sta_b.shape[-2] != s1:
            old = sta_b.shape[-1]
            if old == s1 * 2:
                # pool 2x2 blocks by OR to downsample
                sta_b = sta_b.reshape(sta_b.shape[0], sta_b.shape[1], s1, 2, s1, 2).any(dim=(-1, -3))
            else:
                raise ValueError(f"sta grid {sta_b.shape[-2:]} does not match s1={s1} (or 2*s1)")

        # Now expand head dim if needed
        if sta_b.shape[1] == 1:
            sta_b = sta_b.expand(B, h, s1, s1)
        elif sta_b.shape[1] != h:
            raise ValueError(f"sta head dim {sta_b.shape[1]} must be 1 or h={h}")

        mask = mask | sta_b

    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)

    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=BLOCK,
        mask_mod=None,
    )
