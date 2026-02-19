# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License
import torch
from PIL import Image
from tqdm import tqdm
from .models.utils import fast_sta_nabla
import torchvision.transforms.functional as F
from math import sqrt
from typing import Sequence, Union
from musubi_tuner.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from blissful_tuner.latent_preview import LatentPreviewer
from blissful_tuner.guidance import apply_zerostar_scaling
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

_GLOBAL_DTYPE = torch.bfloat16


def set_global_dtype(dtype: torch.dtype):
    global _GLOBAL_DTYPE
    if isinstance(dtype, torch.dtype):
        from musubi_tuner.kandinsky5.models.nn import set_global_dtype_nn

        _GLOBAL_DTYPE = dtype
        set_global_dtype_nn(dtype)
        logger.info(f"Global dtype updated to {_GLOBAL_DTYPE}!")
    else:
        raise ValueError("Global dtype must be a torch.dtype!")


def resize_image(image, max_area, divisibility=16):
    h, w = image.shape[2:]
    area = h * w
    k = min(1.0, sqrt(max_area / area))
    new_h = int(round((h * k) / divisibility) * divisibility)
    new_w = int(round((w * k) / divisibility) * divisibility)
    new_h = max(divisibility, new_h)
    new_w = max(divisibility, new_w)
    return F.resize(image, (new_h, new_w)), k


def _to_pil(image):
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image
    raise ValueError(f"unknown image type: {type(image)}")


def get_reference_latents(
    image: Union[str, Image.Image, Sequence],
    vae,
    device,
    max_area,
    divisibility,
    i2v_mode: str = "first",
    size_override: tuple[int, int] = None,
    is_flux_vae: bool = False,
):
    """
    Returns reference PIL (first element), stacked reference latents [N, H, W, C], and resize scale.
    Supports single image or list/tuple for first+last conditioning.
    """
    if isinstance(image, (list, tuple)):
        pil_images = [_to_pil(im) for im in image]
    else:
        pil_images = [_to_pil(image)]

    # resize target from the first image to keep spatial shape consistent across references
    image_tensor = F.pil_to_tensor(pil_images[0]).unsqueeze(0)
    if size_override is None:
        image_tensor, k = resize_image(image_tensor, max_area=max_area, divisibility=divisibility)
    else:
        o_height, o_width = size_override
        image_tensor = F.resize(image_tensor, [o_height, o_width], interpolation=F.InterpolationMode.BICUBIC)
        k = None
    target_hw = image_tensor.shape[2:]

    latents = []
    target_dtype = getattr(vae, "dtype", torch.float16)
    for pil in pil_images:
        tensor = F.pil_to_tensor(pil).unsqueeze(0)
        tensor = F.resize(tensor, target_hw)
        tensor = tensor / 127.5 - 1.0
        with torch.no_grad():
            if not is_flux_vae:
                tensor = tensor.to(device=device, dtype=target_dtype).transpose(0, 1).unsqueeze(0)
                try:
                    enc_out = vae.encode(tensor, opt_tiling=False)
                except TypeError:
                    enc_out = vae.encode(tensor)
                lat_image = enc_out.latent_dist.sample().squeeze(0).permute(1, 2, 3, 0)  # 1, C, 1, H, W -> 1, H, W, C
            else:
                tensor = tensor.to(device, target_dtype)
                enc_out = vae.encode(tensor)
                lat_image = enc_out.latent_dist.sample().permute(0, 2, 3, 1)  # 1, C, H, W -> 1, H, W, C
            lat_image = lat_image * vae.config.scaling_factor
            latents.append(lat_image)

    latents = torch.stack(latents, dim=0) if len(latents) > 1 else latents[0]

    # If caller requested first_last but only one image provided, duplicate to keep indices valid downstream.
    if i2v_mode == "first_last" and latents.dim() == 3:
        latents = torch.stack([latents, latents], dim=0)

    return pil_images[0], latents, k


def get_first_frame_from_image(image, vae, device, max_area, divisibility, size_override=None, is_flux_vae=False):
    """Backward-compatible helper: returns a single-frame latent and scale."""
    pil, latents, k = get_reference_latents(
        image, vae, device, max_area, divisibility, i2v_mode="first", size_override=size_override, is_flux_vae=is_flux_vae
    )
    if latents.dim() == 4 and latents.shape[0] > 1:
        latents = latents[0]
    return pil, latents, k


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(
            T, H // 8, W // 8, conf.model.attention.wT, conf.model.attention.wH, conf.model.attention.wW, device=device
        )
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params


def adaptive_mean_std_normalization(source, reference):
    source_mean = source.mean(dim=(1, 2, 3), keepdim=True)
    source_std = source.std(dim=(1, 2, 3), keepdim=True)
    # magic constants - limit changes in latents
    clump_mean_low = 0.05
    clump_mean_high = 0.1
    clump_std_low = 0.1
    clump_std_high = 0.25

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / source_std
    normalized = normalized * reference_std + reference_mean

    return normalized


def normalize_first_frame(latents, reference_frames=5, clump_values=False):
    latents_copy = latents.clone()
    samples = latents_copy

    if samples.shape[0] <= 1:
        return (latents, "Only one frame, no normalization needed")
    nFr = 4
    first_frames = samples[:nFr]
    reference_frames_data = samples[nFr : nFr + min(reference_frames, samples.shape[0] - 1)]

    # logger.info("First frame stats - Mean:", first_frames.mean(dim=(1,2,3)), "Std: ", first_frames.std(dim=(1,2,3)))
    # logger.info(f"Reference frames stats - Mean: {reference_frames_data.mean().item():.4f}, Std: {reference_frames_data.std().item():.4f}")

    normalized_first = adaptive_mean_std_normalization(first_frames, reference_frames_data)
    if clump_values:
        min_val = reference_frames_data.min()
        max_val = reference_frames_data.max()
        normalized_first = torch.clamp(normalized_first, min_val, max_val)

    samples[:nFr] = normalized_first

    return samples


@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
    blissful_args=None,
    cur_step=None,
):
    do_cfg_for_step = True  # Default true so behavior only altered if blissful args present
    do_zero_init = do_zero_scale = False
    if (
        blissful_args is not None and blissful_args["args"] is not None
    ):  # if args is None we were called from other than generation so skip blissful
        scale_per_step = blissful_args["scale_per_step"]
        args = blissful_args["args"]
        if args.cfgzerostar_scaling:
            do_zero_scale = True
        if args.cfgzerostar_init_steps != -1:
            do_zero_init = True
        if args.cfg_schedule is not None and scale_per_step is not None and cur_step is not None:  # Shield
            do_cfg_for_step = (cur_step + 1) in scale_per_step
            if do_cfg_for_step:
                guidance_weight = scale_per_step[cur_step + 1]

    with torch._dynamo.utils.disable_cache_limit():
        pred_velocity = dit(
            x,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
        )
        if abs(guidance_weight - 1.0) > 1e-6 and do_cfg_for_step:
            uncond_pred_velocity = dit(
                x,
                null_text_embeds["text_embeds"],
                null_text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                null_text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=null_attention_mask,
            )
            if do_zero_scale:
                pred_out = apply_zerostar_scaling(pred_velocity, uncond_pred_velocity, guidance_weight)
            else:
                pred_out = uncond_pred_velocity + guidance_weight * (pred_velocity - uncond_pred_velocity)
            if do_zero_init and cur_step <= args.cfgzerostar_init_steps - 1:
                pred_out *= args.cfgzerostar_multiplier
            pred_velocity = pred_out
    return pred_velocity


@torch.no_grad()
def decode_latents(latent_visual, vae, device="cuda", batch_size=1, num_frames=None, mode="hv"):
    """latent_visual: [B*F, H, W, C] -> returns uint8 [B, F, H, W, 3]"""
    b_times_f, h, w, c = latent_visual.shape
    if num_frames is None:
        num_frames = b_times_f // batch_size

    latent_visual = latent_visual.reshape(batch_size, num_frames, h, w, c)
    latent_visual = latent_visual.to(device=device, dtype=vae.dtype)

    # [B, F, H, W, C] -> [B, C, F, H, W]
    latents_5d = (latent_visual / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)  # HV VAE compatible

    if mode == "fx":  # Flux VAE compatible
        B, C, F, Hlat, Wlat = latents_5d.shape
        latents_2d = latents_5d.permute(0, 2, 1, 3, 4).reshape(B * F, C, Hlat, Wlat)

        decoded_2d = vae.decode(latents_2d).sample  # [B*F, 3, Himg, Wimg]

        Himg, Wimg = decoded_2d.shape[-2], decoded_2d.shape[-1]
        images = decoded_2d.reshape(B, F, decoded_2d.shape[1], Himg, Wimg).permute(0, 2, 1, 3, 4)
    else:
        images = vae.decode(latents_5d).sample

    # [B, 3, F, H, W] -> [B, F, H, W, 3]
    images = images.permute(0, 2, 3, 4, 1)
    images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    return images


def i2v_slice(
    i2v_frames: torch.Tensor,
    latent: torch.Tensor,
    visual_cond_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Slice first and optionally last frames into provided latent at positions 0, -1, and optionally update vcond mask"""
    if i2v_frames is not None:
        i2vf = i2v_frames.to(device=latent.device, dtype=latent.dtype)

        # Normalize to [F,H,W,C]
        if i2vf.dim() == latent.dim() - 1:
            i2vf = i2vf.unsqueeze(0)

        if i2vf.shape[0] not in (1, 2):
            raise ValueError(f"Expected 1 or 2 conditioning frames, got {i2vf.shape[0]}")

        # Always clamp the first frame
        latent[:1] = i2vf[0:1]
        if visual_cond_mask is not None:
            visual_cond_mask[:1] = 1

        # If we have a second frame, clamp the last frame too
        if i2vf.shape[0] == 2:
            latent[-1:] = i2vf[1:2]
            if visual_cond_mask is not None:
                visual_cond_mask[-1:] = 1
    return latent, visual_cond_mask


@torch.no_grad()
def generate_sample_latents_only(
    shape,
    dit,
    text_embeds,
    pooled_embed,
    attention_mask,
    null_text_embeds=None,
    null_pooled_embed=None,
    null_attention_mask=None,
    i2v_frames=None,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    seed=6554,
    device="cuda",
    conf=None,
    progress=False,
    i2v_mode=None,  # unused; kept for call-site compatibility
    blissful_args=None,
    image_edit=False,
):
    """Minimal sampler that returns latents only (no VAE decode)."""
    bs, duration, height, width, dim = shape
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=_GLOBAL_DTYPE)
    seq_len = text_embeds.shape[0] if text_embeds.dim() == 2 else text_embeds.shape[1]

    if null_text_embeds is None:
        null_text_embeds = torch.zeros_like(text_embeds)

    null_seq_len = null_text_embeds.shape[0] if null_text_embeds.dim() == 2 else null_text_embeds.shape[1]
    if null_pooled_embed is None:
        null_pooled_embed = torch.zeros_like(pooled_embed)

    text_embeds = text_embeds.to(device=device)
    null_text_embeds = null_text_embeds.to(device=device)
    pooled_embed = pooled_embed.to(device=device)
    null_pooled_embed = null_pooled_embed.to(device=device)

    text_dict = {"text_embeds": text_embeds, "pooled_embed": pooled_embed}
    null_text_dict = {"text_embeds": null_text_embeds, "pooled_embed": null_pooled_embed}
    if blissful_args is not None:
        blissful_args["first_noise"] = img

    if image_edit:
        if dit.instruct_type == "channel":
            image = None if i2v_frames is None else i2v_frames[0:1]
            if image is not None:
                edit_latent = torch.cat([image, torch.ones_like(img[..., :1])], -1)
            else:
                edit_latent = torch.cat([torch.zeros_like(img), torch.zeros_like(img[..., :1])], -1)
            img = torch.cat([img, edit_latent], dim=-1)
    # Shape/patch sanity guard: visual grid must be divisible by patch sizes
    ps_t, ps_h, ps_w = conf.model.dit_params.patch_size
    if (height % ps_h) != 0 or (width % ps_w) != 0 or (duration % ps_t) != 0:
        raise ValueError(
            f"Invalid visual shape for patch_size {ps_t, ps_h, ps_w}: frames={duration}, height={height}, width={width}"
        )

    visual_rope_pos = [
        torch.arange(duration, device=device),
        torch.arange(height // conf.model.dit_params.patch_size[1], device=device),
        torch.arange(width // conf.model.dit_params.patch_size[2], device=device),
    ]
    text_rope_pos = torch.arange(seq_len, device=device)
    null_text_rope_pos = torch.arange(null_seq_len, device=device)

    latents = generate(
        dit,
        device,
        img,
        num_steps,
        text_dict,
        null_text_dict,
        visual_rope_pos,
        text_rope_pos,
        null_text_rope_pos,
        guidance_weight,
        scheduler_scale,
        i2v_frames,
        conf,
        progress=progress,
        seed=seed,
        tp_mesh=None,
        attention_mask=attention_mask,
        null_attention_mask=null_attention_mask,
        blissful_args=blissful_args,
    )
    if i2v_frames is not None and not image_edit:
        logger.info("I2V post processing!")
        latents, _ = i2v_slice(i2v_frames, latents)
        latents = normalize_first_frame(latents)
    return latents


@torch.no_grad()
def generate(
    model,
    device,
    img,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    i2v_frames,
    conf,
    progress=False,
    seed=6554,
    tp_mesh=None,
    attention_mask=None,
    null_attention_mask=None,
    blissful_args=None,
):
    args = blissful_args["args"] if blissful_args is not None else None
    sparse_params = get_sparse_params(conf, {"visual": img}, device)

    # Setup scheduler
    if args is None or args.scheduler == "default":
        logger.info(f"Using default Euler scheduler with shift {scheduler_scale}")
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
        timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)
        scheduler = None
    else:
        logger.info(f"Using DPM++ scheduler with shift {scheduler_scale}!")
        scheduler = FlowDPMSolverMultistepScheduler(
            shift=scheduler_scale,
            use_dynamic_shifting=False,
            algorithm_type="dpmsolver++",
        )
        scheduler.set_timesteps(args.steps, device=device)
        scheduler.set_begin_index(0)
        timesteps = scheduler.sigmas.to(device)

    # Setup previewer
    if args is not None and args.preview_latent_every:
        previewer = LatentPreviewer(
            args,
            original_latents=blissful_args["first_noise"],
            scheduler=scheduler,
            device=device,
            dtype=_GLOBAL_DTYPE,
            model_type="k5" if "2i-" not in args.task else "k5_flux",
        )
        previewer.noise_remain = 1.0000
        if scheduler is None:
            previewer.sigmas = timesteps

    for i, (timestep, timestep_diff) in enumerate(tqdm(zip(timesteps[:-1], torch.diff(timesteps)), total=len(timesteps) - 1)):
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros([*img.shape[:-1], 1], dtype=img.dtype, device=img.device)
            img, visual_cond_mask = i2v_slice(i2v_frames, img, visual_cond_mask)
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img

        pred_velocity = get_velocity(
            model,
            model_input,
            timestep.unsqueeze(0),
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
            blissful_args=blissful_args,
            cur_step=i,
        )
        latent = img[..., : pred_velocity.shape[-1]]  # Slice off any potential extra ti2i channels else does nothing
        if args is None or args.scheduler == "default":
            latent += timestep_diff * pred_velocity
        else:
            latent = scheduler.step(pred_velocity, timestep, latent, return_dict=False)[0]

        if args is not None and args.preview_latent_every:
            previewer.noise_remain += timestep_diff  # Diff is negative so add it to decrease noise_remain
            if (i + 1) % args.preview_latent_every == 0 and i < args.steps:
                previewer.preview(latent)
        img[..., : pred_velocity.shape[-1]] = latent  # potentially slice back in updated latent for next loop, else does nothing
    return latent
