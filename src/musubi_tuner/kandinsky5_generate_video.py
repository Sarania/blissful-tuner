import argparse
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torchvision.transforms.functional import pil_to_tensor
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from datetime import datetime as datetime
import time
from rich_argparse import RichHelpFormatter
from musubi_tuner.kandinsky5.configs import TASK_CONFIGS, AttentionConfig
from musubi_tuner.kandinsky5.generation_utils import (
    generate_sample_latents_only,
    decode_latents,
    get_first_frame_from_image,
    set_global_dtype,
    _to_pil,
)
from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
from musubi_tuner.kandinsky5_train_network import Kandinsky5NetworkTrainer
from musubi_tuner.hv_train_network import clean_memory_on_device
from musubi_tuner.networks import lora_kandinsky

from blissful_tuner.utils import ensure_dtype_form
from blissful_tuner.blissful_core import add_blissful_k5_args, parse_blissful_args
from blissful_tuner.guidance import parse_scheduled_cfg
from blissful_tuner.common_extensions import save_media_advanced, prepare_metadata
from blissful_tuner.blissful_logger import BlissfulLogger


logger = BlissfulLogger(__name__, "green")


def get_time_flag():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def _get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kandinsky5 sampling (mirrors training sampler, no training)", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--task", type=str, default="k5-pro-t2v-5s-sd", choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for generation")
    parser.add_argument("--i", "--image", dest="image", type=str, default=None, help="Init image path for i2v or image edit")
    parser.add_argument(
        "--image_last", type=str, default=None, help="Optional last-frame image path for i2v first_last conditioning"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Folder to save outputs to")
    parser.add_argument("--width", type=int, default=None, help="Requested width of generated output. Default depends on task.")
    parser.add_argument("--height", type=int, default=None, help="Requested height of generated output. Default depends on task.")
    parser.add_argument("--frames", type=int, default=None, help="Output length in latent frames, exclusive of '--video_length'")
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Output length in pixel frames, exclusive of '--frames' and will be rounded up to fit 4n + 1 if necessary. Use 1 for images or image tasks.",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of inference steps, default depends on task but is often 50"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for classifier free guidance. Default is 5.0 for video tasks and 3.5 for image tasks",
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=None,
        help="Like flow shift for other models, alters timestep distribution. Default depends on task but is often 10.0 for videos and 3.0 for images.",
    )
    parser.add_argument(
        "--flow_shift", type=float, default=None, help="Same as --scheduler_scale, for convenience. Don't provide both."
    )
    parser.add_argument("--seed", type=str, default=None, help="Seed for RNG. Default is random.")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for inference. Default is CUDA if available else CPU"
    )
    parser.add_argument("--dit", type=str, required=True, help="Path to diffusion transformer to inference")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Path to VAE for encode/decode. Use HunyuanVae for video and Flux vae for image tasks.",
    )
    parser.add_argument("--text_encoder_qwen", type=str, required=True, help="Path to QwenVL2.5_7B text encoder")
    parser.add_argument("--text_encoder_clip", type=str, required=True, help="Path to CLIP text encoder")
    parser.add_argument(
        "--dit_dtype",
        type=str,
        default=None,
        choices=[None, "bfloat16", "float16", "float32"],
        help="Dtype to use for DiT weights. Default is just use them as is unless fp8 is enabled.",
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to use for VAE weights. Default of 'bfloat16' is recommemnded.",
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of transformer blocks to offload to CPU to save VRAM. Default is 0, max depends on which model. Saves VRAM",
    )
    parser.add_argument(
        "--fp8_scaled", action="store_true", help="Use fp8 scaled quantization to reduce size of DiT and save memory and VRAM"
    )
    parser.add_argument(
        "--fp8_fast",
        action="store_true",
        help="Only available with `--fp8_scaled`, use fast fp8 math available on Ada Lovelace and later Nvidia GPUs",
    )
    parser.add_argument("--disable_numpy_memmap", action="store_true")
    parser.add_argument("--sdpa", action="store_true", help="use SDPA for visual attention")
    parser.add_argument("--flash_attn", action="store_true", help="use FlashAttention 2 for visual attention")
    parser.add_argument("--flash3", action="store_true", help="use FlashAttention 3 for visual attention")
    parser.add_argument("--sage_attn", action="store_true", help="use SageAttention for visual attention")
    parser.add_argument("--xformers", action="store_true", help="use xformers for visual attention")
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s) to merge for inference")
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s), align with lora_weight order"
    )
    parser.add_argument("--fps", type=int, default=24, help="FPS for output video, 24 is default for K5")
    parser = add_blissful_k5_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    args = parse_blissful_args(args)
    if args.fp8_fast and not args.fp8_scaled:
        raise ValueError("`--fp8_fast` requires `--fp8_scaled` but it is not enabled!")

    if args.flow_shift and not args.scheduler_scale:
        args.scheduler_scale = args.flow_shift
    elif args.scheduler_scale and args.flow_shift:
        logger.warning(
            "'--scheduler_scale' and '--flow_shift' control the same value, please only provide one. For now, we'll use the value of `--scheduler_scale` ({args.scheduler_scale}) and proceed."
        )

    if sum([args.text_encoder_cpu, args.quantized_qwen, args.text_encoder_auto]) > 1:
        raise ValueError(
            "Only one of '--quantized_qwen', '--text_encoder_cpu', '--text_encoder_auto' may be used at a time but received more than that!"
        )

    if args.frames and args.video_length:
        raise ValueError("Only one of '--frames' and '--video_length' is allowed but recieved both!")

    if args.video_length is not None:
        original = args.video_length
        corrected = 4 * ((original - 1 + 3) // 4) + 1
        if corrected != original:
            logger.warning(f"video_length {original} is invalid; rounding up to {corrected} (4n + 1 required)")
        args.video_length = corrected
        args.frames = (corrected - 1) // 4 + 1
    elif args.frames is not None:
        args.video_length = ((args.frames - 1) * 4) + 1
    elif "-t2i-" in args.task or "-i2i-" in args.task:
        args.frames = args.video_length = frames = 1
    else:
        raise ValueError(
            "Neither --frames nor --video_length was provided and mode is not t2i or i2i so we don't know how many frames to make!"
        )

    if args.compile:
        logger.info("Enabling torch.compile!")
        from musubi_tuner.kandinsky5.models.nn import activate_compile

        activate_compile()

    args.vae_dtype = ensure_dtype_form(args.vae_dtype, out_form="torch")
    task_conf = TASK_CONFIGS[args.task]
    if not args.use_nabla_attn:
        if task_conf.attention.type != "flash":
            logger.info("Overriding attention backend to traditional(Flash/Sage/Xformers)!")
            task_conf.attention = AttentionConfig(type="flash", chunk=False, causal=False, local=False, glob=False, window=3)
    else:
        if args.compile:
            logger.info("Overriding attention backend to NABLA!")
            task_conf.attention = AttentionConfig(
                type="nabla",
                chunk=False,
                causal=False,
                local=False,
                glob=False,
                window=3,
                method="topcdf",
                P=args.nabla_p,
                add_sta=True,
                wT=11,
                wH=3,
                wW=3,
            )
        else:
            raise ValueError("Requested NABLA but not compile, NABLA uses flex_attention on the backend and requires --compile")

    device = _get_device(args.device)

    if args.advanced_i2v and task_conf.attention.type == "nabla":
        raise ValueError("Cannot allow '--advanced_i2v' when NABLA attention is enabled!")

    width = args.width or task_conf.resolution
    height = args.height or task_conf.resolution
    # round width and height to multiples of 16
    width = (width // 16) * 16
    height = (height // 16) * 16
    frames = args.frames if args.frames is not None else (5 if task_conf.dit_params.visual_cond else 1)
    i2v_mode = "first_last" if args.image_last else "first"
    steps = args.steps or task_conf.num_steps
    args.steps = steps  # Previewer need

    guidance = args.guidance_scale if args.guidance_scale is not None else task_conf.guidance_weight
    scheduler_scale = args.scheduler_scale if args.scheduler_scale is not None else (task_conf.scheduler_scale or 1.0)
    latent_h = max(1, height // 8)
    latent_w = max(1, width // 8)
    shape = (1, frames, latent_h, latent_w, task_conf.dit_params.in_visual_dim)
    logger.info(
        f"WHF: {width}x{height}x{args.video_length}, Latent WHF: {latent_w}x{latent_h}x{frames}, Steps: {steps}, Guidance: {guidance}, Shift: {scheduler_scale}"
    )
    # Resolve paths
    dit_path = args.dit or task_conf.checkpoint_path
    vae_path = args.vae or task_conf.vae.checkpoint_path
    qwen_path = args.text_encoder_qwen or task_conf.text.qwen_checkpoint
    clip_path = args.text_encoder_clip or task_conf.text.clip_checkpoint

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Build trainer (reuse training loaders)
    trainer = Kandinsky5NetworkTrainer()
    trainer.task_conf = task_conf
    trainer.blocks_to_swap = args.blocks_to_swap
    trainer._text_encoder_qwen_path = args.text_encoder_qwen
    trainer._text_encoder_clip_path = args.text_encoder_clip
    trainer._vae_checkpoint_path = vae_path
    original_base_name = None
    if args.decode_from_latent:
        logger.info(f"Loading latent from {args.decode_from_latent}")
        latents = load_file(args.decode_from_latent)["latent"]
        with safe_open(args.decode_from_latent, framework="pt") as f:
            metadata = f.metadata() if not args.no_metadata else None
        logger.info(f"Prepared metadata: {metadata}")
        frames = args.frames = len(latents)
        args.video_length = ((args.frames - 1) * 4) + 1
        vae = trainer._load_vae_for_sampling(args, device=device)
        logger.info(f"Decoding {args.frames} latent frames into {args.video_length} pixel frames!")
        images = decode_latents(latents, vae, device=device, batch_size=shape[0], mode="hv" if "2i-" not in args.task else "fx")
        original_base_name = os.path.splitext(os.path.basename(args.decode_from_latent))[0].replace("_latent", "")
    else:
        metadata = prepare_metadata(args) if not args.no_metadata else None
        logger.info(f"Prepared metadata: {metadata}")
        # --- Stage 1: text encoder only ---
        text_embedder_conf = SimpleNamespace(
            qwen=SimpleNamespace(checkpoint_path=qwen_path, max_length=task_conf.text.qwen_max_length),
            clip=SimpleNamespace(checkpoint_path=clip_path, max_length=task_conf.text.clip_max_length),
        )
        text_embedder = get_text_embedder(
            text_embedder_conf,
            device="cpu" if args.text_encoder_cpu else device,
            quantized_qwen=args.quantized_qwen if not args.text_encoder_cpu else False,
            qwen_auto=args.text_encoder_auto,
        )
        image_edit = args.image is not None and "-i2i-" in args.task
        te_image = None if not image_edit else pil_to_tensor(_to_pil(args.image))
        neg_text = args.negative_prompt or "low quality, bad quality"
        enc_out, _ = text_embedder.encode(
            [args.prompt],
            te_image,
            type_of_content=("video" if frames > 1 else "image" if not image_edit else "image_edit"),
            use_system=True,
        )
        neg_out, _ = text_embedder.encode(
            [neg_text],
            te_image,
            type_of_content=("video" if frames > 1 else "image" if not image_edit else "image_edit"),
            use_system=True,
        )
        text_embeds = enc_out["text_embeds"].to("cpu")
        pooled_embed = enc_out["pooled_embed"].to("cpu")
        null_text_embeds = neg_out["text_embeds"].to("cpu")
        null_pooled_embed = neg_out["pooled_embed"].to("cpu")

        try:
            text_embedder.to("cpu")
        except Exception:
            pass
        del text_embedder
        clean_memory_on_device(device)

        scale_per_step = None
        if args.cfg_schedule is not None:
            scale_per_step = parse_scheduled_cfg(args.cfg_schedule, steps, guidance)
            included_steps = sorted(scale_per_step.keys())
            step_str = ", ".join(f"{step}:{scale_per_step[step]}" for step in included_steps)
            logger.info(f"CFG Schedule: {step_str}")
            logger.info(f"Total CFG steps: {len(included_steps)}")
        else:
            logger.info("Full CFG enabled!")
        if args.tf32_mode:
            logger.info("Enabling TF32 mode!")
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"

        # Prepare I2V
        i2v_frames = None
        # Optional init image(s) -> latent first/last frames (i2v-style). Requires temporary VAE load.
        if args.image:
            is_flux_vae = "-i2i-" in args.task
            vae_for_encode = trainer._load_vae_for_sampling(args, device=device)
            max_area = 2048 * 2048 if args.advanced_i2v else 512 * 768 if int(task_conf.resolution) == 512 else 1024 * 1024
            divisibility = 16 if args.advanced_i2v or task_conf.attention.type != "nabla" else 128
            size_override = None if not args.advanced_i2v else (height, width)
            # Always encode the first image
            _, lat_image_first, _ = get_first_frame_from_image(
                args.image,
                vae_for_encode,
                device,
                max_area=max_area,
                divisibility=divisibility,
                size_override=size_override,
                is_flux_vae=is_flux_vae,
            )
            frame_list = [lat_image_first[:1]]
            # Optionally encode the last image
            if args.image_last:
                _, lat_image_last, _ = get_first_frame_from_image(
                    args.image_last,
                    vae_for_encode,
                    device,
                    max_area=max_area,
                    divisibility=divisibility,
                    size_override=size_override,
                    is_flux_vae=is_flux_vae,
                )
                frame_list.append(lat_image_last[:1])
            i2v_frames = torch.cat(frame_list, dim=0)
            # If the init image was resized by the encoder, match sampling shape to it.
            if i2v_frames is not None:
                latent_h = int(i2v_frames.shape[1])
                latent_w = int(i2v_frames.shape[2])
                old_w = width
                old_h = height
                width = latent_w * 8
                height = latent_h * 8
                shape = (1, frames, latent_h, latent_w, task_conf.dit_params.in_visual_dim)
                if old_w != width or old_h != height:
                    logger.warning(f"I2VI updated resolution (W*H) to: {width}x{height}, latent: {latent_w}x{latent_h}")
            vae_for_encode.to("cpu")
            del vae_for_encode
            clean_memory_on_device(device)
        dit_weight_dtype = None if args.fp8_scaled else ensure_dtype_form(args.dit_dtype, out_form="torch")
        conf_ns = SimpleNamespace(model=task_conf, metrics=SimpleNamespace(scale_factor=task_conf.scale_factor))

        # --- Stage 2: load DiT, sample latents ---
        loader_args = SimpleNamespace(
            fp8_scaled=args.fp8_scaled if not args.lora_weight else None,
            fp8_fast=args.fp8_fast if not args.lora_weight else None,
            blocks_to_swap=args.blocks_to_swap,
            disable_numpy_memmap=args.disable_numpy_memmap,
            override_dit=None,
            sdpa=args.sdpa,
            flash_attn=args.flash_attn,
            flash3=args.flash3,
            sage_attn=args.sage_attn,
            xformers=args.xformers,
        )
        accel_stub = SimpleNamespace(device=device)
        dit = trainer.load_transformer(
            accelerator=accel_stub,
            args=loader_args,
            dit_path=dit_path,
            attn_mode=task_conf.attention.type,
            split_attn=False,
            loading_device=device,
            dit_weight_dtype=dit_weight_dtype,
        )
        dit.eval()
        dit.requires_grad_(False)

        # Merge LoRA weights before any casting/offloading.
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            for idx, lora_path in enumerate(args.lora_weight):
                mult = args.lora_multiplier[idx] if args.lora_multiplier and len(args.lora_multiplier) > idx else 1.0
                lora_sd = load_file(lora_path)
                net = lora_kandinsky.create_arch_network_from_weights(mult, lora_sd, unet=dit, for_inference=True)
                net.merge_to(None, dit, lora_sd, device=dit.device if hasattr(dit, "device") else device, non_blocking=True)
            clean_memory_on_device(device)
            if args.fp8_scaled:
                state_dict = dit.state_dict()

                # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
                move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
                state_dict = dit.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

                info = dit.load_state_dict(state_dict, strict=True, assign=True)
                logger.info(f"Loaded FP8 optimized weights: {info}")

        if dit_weight_dtype is not None and dit_weight_dtype != torch.float32:  # Ensure dtype compliance
            logger.info(f"Casting DiT to {dit_weight_dtype}")
            dit.to(dit_weight_dtype)
            dit.dtype = dit_weight_dtype

        if args.blocks_to_swap > 0:
            dit.enable_block_swap(args.blocks_to_swap, device, supports_backward=False, use_pinned_memory=False)
            dit.move_to_device_except_swap_blocks(device)
            dit.prepare_block_swap_before_forward()
        else:
            dit.to(device)

        autocast_dtype = torch.bfloat16

        blissful_args = {"scale_per_step": scale_per_step, "args": args}

        if args.fp16_fast:
            logger.info("Enabling fp16 accumulation and switching math dtype to float16 for fp16_fast.")
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            autocast_dtype = torch.float16

        if autocast_dtype is not None:
            set_global_dtype(autocast_dtype)

        if args.cfgzerostar_scaling or args.cfgzerostar_init_steps != -1:
            logger.info(
                f"Using CFGZero* - Scaling: {args.cfgzerostar_scaling}; Zero init steps: {'None' if args.cfgzerostar_init_steps == -1 else args.cfgzerostar_init_steps}"
            )

        logger.info(f"DiT dtype: {dit.dtype}; Autocast dtype: {autocast_dtype}")

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None), torch.no_grad():
            latents = generate_sample_latents_only(
                shape=shape,
                dit=dit,
                text_embeds=text_embeds,
                pooled_embed=pooled_embed,
                attention_mask=None,
                null_text_embeds=null_text_embeds,
                null_pooled_embed=null_pooled_embed,
                null_attention_mask=None,
                i2v_frames=i2v_frames,
                num_steps=steps,
                guidance_weight=guidance,
                scheduler_scale=scheduler_scale,
                seed=args.seed,
                device=device,
                conf=conf_ns,
                progress=True,
                i2v_mode=i2v_mode,
                blissful_args=blissful_args,
            )
        # free DiT
        dit.to("cpu")
        del dit
        clean_memory_on_device(device)
        time_flag = get_time_flag()

        # --- Stage 3: load VAE, decode ---
        if args.output_type in ["latent", "both"]:
            latent_path = f"{args.save_path}/{time_flag}_{args.seed}_latent.safetensors"
            logger.info(f"Save latent to {latent_path}!")
            sd = {"latent": latents}
            save_file(sd, latent_path, metadata=metadata)
        if args.output_type in ["video", "both"]:
            vae = trainer._load_vae_for_sampling(args, device=device)
            images = decode_latents(
                latents, vae, device=device, batch_size=shape[0], num_frames=frames, mode="hv" if "2i-" not in args.task else "fx"
            )
            try:
                vae.to("cpu")
            except Exception:
                pass
            del vae
        clean_memory_on_device(device)
    # Save
    if args.output_type != "latent":  # images incoming is  B, F, H, W, C
        video_tensor = images.permute(0, 4, 1, 2, 3).float() / 255.0  # B, C, F, H, W
        video_tensor = video_tensor.cpu()
        video_path = (
            f"{args.save_path}/{time_flag}_{args.seed}.mp4"
            if not args.decode_from_latent
            else f"{args.save_path}/{original_base_name}.mp4"
        )
        save_media_advanced(
            video_tensor, video_path, args, metadata=metadata
        )  # Handles single frame case internally choosing between vid/image
        if images.shape[1] > 1:  # If we have a vid, save first (and potentially last) frame too
            first_frame_path = os.path.splitext(video_path)[0] + "_first.png"
            frame = images[:, 0:1]  # Slice out first, B, 1, H, W, C
            frame = frame.permute(0, 4, 1, 2, 3).float() / 255.0  # Prepare shape for SMA
            frame = frame.cpu()
            save_media_advanced(frame, first_frame_path, args, metadata=metadata)
            if args.save_last_frame:
                last_frame_path = os.path.splitext(video_path)[0] + "_last.png"
                frame = images[:, -1:]  # Slice out last
                frame = frame.permute(0, 4, 1, 2, 3).float() / 255.0  # Prepare shape
                frame = frame.cpu()
                save_media_advanced(frame, last_frame_path, args, metadata=metadata)


if __name__ == "__main__":
    main()
