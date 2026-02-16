import argparse
import json
from types import SimpleNamespace
from typing import Optional
import random
import torch
from accelerate import Accelerator, init_empty_weights
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_KANDINSKY5, ARCHITECTURE_KANDINSKY5_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
    load_prompts,
)
from musubi_tuner.kandinsky5.configs import TASK_CONFIGS, TaskConfig
from musubi_tuner.kandinsky5.models.dit import DiffusionTransformer3D, get_dit
from musubi_tuner.kandinsky5.models.vae import build_vae
from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
from musubi_tuner.kandinsky5 import generation_utils
from musubi_tuner.kandinsky5.models.utils import fast_sta_nabla
from musubi_tuner.kandinsky5.generation_utils import get_first_frame_from_image
from musubi_tuner.kandinsky5.models import nn as k5_nn
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

from blissful_tuner.blissful_logger import BlissfulLogger
from blissful_tuner.utils import ensure_dtype_form

logger = BlissfulLogger(__name__, "green")


class Kandinsky5NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.dit_conf = None
        self.task_conf: TaskConfig = None
        self.vae_scaling_factor = 1.0
        self._vae_checkpoint_path: str | None = None
        self._cached_sample_vae = None
        self._text_encoder_qwen_path: str | None = None
        self._text_encoder_clip_path: str | None = None
        self._nabla_mask_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
        self._i2v_training = False
        self._i2v_mode = "first"
        self._control_training = False
        self.visual_cond_prob: float = 1.0

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KANDINSKY5

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KANDINSKY5_FULL

    def handle_model_specific_args(self, args):
        if args.task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{args.task}'. Available: {list(TASK_CONFIGS.keys())}")
        self.task_conf = TASK_CONFIGS[args.task]

        if getattr(args, "force_nabla_attention", False):
            from dataclasses import replace

            self.task_conf = replace(
                self.task_conf,
                attention=replace(
                    self.task_conf.attention,
                    type="nabla",
                    method=getattr(args, "nabla_method", "topcdf"),
                    P=getattr(args, "nabla_P", 0.9),
                    add_sta=getattr(args, "nabla_add_sta", True),
                    wT=getattr(args, "nabla_wT", 11),
                    wH=getattr(args, "nabla_wH", 3),
                    wW=getattr(args, "nabla_wW", 3),
                ),
            )
            logger.info(
                "Forcing nabla attention for training: "
                f"method={self.task_conf.attention.method}, P={self.task_conf.attention.P}, "
                f"wT={self.task_conf.attention.wT}, wH={self.task_conf.attention.wH}, wW={self.task_conf.attention.wW}, "
                f"add_sta={self.task_conf.attention.add_sta}"
            )
        self.dit_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        self._i2v_training = "i2v" in args.task
        if self._i2v_training:
            logger.info("I2V training mode enabled")
        self._i2v_mode = getattr(args, "i2v_mode", "first") or "first"
        self._control_training = False
        self.visual_cond_prob = float(getattr(args, "visual_cond_prob", 1.0) or 0.0)
        if self.visual_cond_prob < 0.0 or self.visual_cond_prob > 1.0:
            logger.warning(f"visual_cond_prob {self.visual_cond_prob} out of [0,1]; clamping.")
            self.visual_cond_prob = min(1.0, max(0.0, self.visual_cond_prob))
        self.default_guidance_scale = self.task_conf.guidance_weight
        # Text token padding is always enabled to mirror the working inference path.
        self._text_encoder_qwen_path = getattr(args, "text_encoder_qwen", None)
        self._text_encoder_clip_path = getattr(args, "text_encoder_clip", None)
        self._vae_checkpoint_path = getattr(args, "vae", None) or self.task_conf.vae.checkpoint_path

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        return self._control_training

    def _build_sparse_params(self, x: torch.Tensor, device: torch.device):
        """Create (and cache) nabla sparse attention params for the current visual grid."""
        attn_conf = getattr(self.task_conf, "attention", None)
        if attn_conf is None or getattr(attn_conf, "type", None) != "nabla":
            return None
        if not self.dit_conf:
            return None

        patch_size = self.dit_conf.get("patch_size", (1, 2, 2))
        # Enforce geometry assumptions required by NABLA/STA masks and fractal flattening.
        if patch_size[0] != 1:
            raise ValueError("NABLA requires temporal patch size == 1 (got patch_size[0] != 1)")

        duration, height, width = x.shape[:3]
        if height % patch_size[1] != 0 or width % patch_size[2] != 0:
            raise ValueError(f"NABLA requires spatial dims divisible by patch_size; got H={height}, W={width}, patch={patch_size}")
        T = duration // patch_size[0]
        H = height // patch_size[1]
        W = width // patch_size[2]
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(f"NABLA requires H//patch and W//patch divisible by 8 for fractal flattening; got H={H}, W={W}")

        # Cache STA masks per (T, H/8, W/8, device) to avoid recomputing every step.
        sta_key = (T, H // 8, W // 8, device)
        sta_mask = self._nabla_mask_cache.get(sta_key)
        if sta_mask is None:
            sta_mask = fast_sta_nabla(
                T,
                H // 8,
                W // 8,
                attn_conf.wT,
                attn_conf.wH,
                attn_conf.wW,
                device=device,
            )
            self._nabla_mask_cache[sta_key] = sta_mask

        return {
            "sta_mask": sta_mask.unsqueeze(0).unsqueeze(0),
            "attention_type": attn_conf.type,
            "to_fractal": True,
            "P": attn_conf.P,
            "wT": attn_conf.wT,
            "wW": attn_conf.wW,
            "wH": attn_conf.wH,
            "add_sta": attn_conf.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(attn_conf, "method", "topcdf"),
        }

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        device = accelerator.device
        latent_w = max(1, width // 8)
        latent_h = max(1, height // 8)
        latent_f = max(1, (frame_count - 1) // 4 + 1)

        text_embeds = sample_parameter["text_embeds"].to(device=device, dtype=dit_dtype)
        pooled_embed = sample_parameter["pooled_embed"].to(device=device, dtype=dit_dtype)
        null_text_embeds = sample_parameter["null_text_embeds"].to(device=device, dtype=dit_dtype)
        null_pooled_embed = sample_parameter["null_pooled_embed"].to(device=device, dtype=dit_dtype)
        seed = sample_parameter.get("seed", random.getrandbits(32))
        scheduler_scale = sample_parameter.get(
            "discrete_flow_shift", self.task_conf.scheduler_scale or 5.0
        )  # Ignore 14.5 default by pulling it ourself and defaulting to task conf
        guidance_scale = cfg_scale or self.task_conf.guidance_weight
        conf_ns = SimpleNamespace(model=self.task_conf, metrics=SimpleNamespace(scale_factor=self.task_conf.scale_factor))
        image_path = sample_parameter.get("image_path", None)
        end_image_path = sample_parameter.get("end_image_path", None)
        i2v_frames = vae_for_sampling = None

        if image_path:
            frame_list = []
            vae_for_sampling = self._load_vae_for_sampling(args, accelerator.device)
            max_area = 512 * 768 if int(self.task_conf.resolution) == 512 else 1024 * 1024
            divisibility = 16 if int(self.task_conf.resolution) == 512 else 128
            _, start_image, _ = get_first_frame_from_image(
                image_path,
                vae_for_sampling,
                accelerator.device,
                max_area=max_area,
                divisibility=divisibility,
            )
            frame_list.append(start_image[:1])
            if end_image_path:
                _, end_image, _ = get_first_frame_from_image(
                    end_image_path,
                    vae_for_sampling,
                    accelerator.device,
                    max_area=max_area,
                    divisibility=divisibility,
                )
                frame_list.append(end_image[:1])
            i2v_frames = torch.cat(frame_list, dim=0)
            vae_for_sampling.to("cpu")  # Save for later
            if i2v_frames is not None:
                latent_h = int(i2v_frames.shape[1])
                latent_w = int(i2v_frames.shape[2])

        shape = (1, latent_f, latent_h, latent_w, self.task_conf.dit_params.in_visual_dim)

        latents = generation_utils.generate_sample_latents_only(
            shape=shape,
            dit=transformer,
            text_embeds=text_embeds,
            pooled_embed=pooled_embed,
            attention_mask=None,
            null_text_embeds=null_text_embeds,
            null_pooled_embed=null_pooled_embed,
            null_attention_mask=None,
            i2v_frames=i2v_frames,
            num_steps=sample_steps,
            guidance_weight=guidance_scale,
            scheduler_scale=scheduler_scale,
            seed=seed,
            device=accelerator.device,
            conf=conf_ns,
            progress=False,
            blissful_args=None,
        )
        vae_for_sampling = self._load_vae_for_sampling(args, accelerator.device) if vae_for_sampling is None else vae_for_sampling
        vae_for_sampling.to(device)
        vae_for_sampling.eval()
        video = generation_utils.decode_latents(
            latents,
            vae_for_sampling,
            device=accelerator.device,
            batch_size=shape[0],
            num_frames=latent_f,
        )
        vae_for_sampling.to("cpu")
        del vae_for_sampling

        video = video.permute(0, 4, 1, 2, 3).float() / 255.0  # B, F, H, W, C ->  B, C, F, H, W
        video = video.cpu()
        clean_memory_on_device(device)
        return video

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_embedder):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    if "negative_prompt" not in prompt_dict:
                        prompt_dict["negative_prompt"] = "low quality, bad quality"
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            toc = "video" if prompt_dict["frame_count"] > 1 else "image"
                            logger.info(f"Caching Text Encoder outputs for {toc} prompt: '{p}'")
                            enc_out, _ = text_embedder.encode([p], type_of_content=toc)
                            sample_prompts_te_outputs[p] = enc_out
            return sample_prompts_te_outputs

        text_embedder_conf = SimpleNamespace(
            qwen=SimpleNamespace(
                checkpoint_path=self._text_encoder_qwen_path or self.task_conf.text.qwen_checkpoint,
                max_length=self.task_conf.text.qwen_max_length,
            ),
            clip=SimpleNamespace(
                checkpoint_path=self._text_encoder_clip_path or self.task_conf.text.clip_checkpoint,
                max_length=self.task_conf.text.clip_max_length,
            ),
        )
        # 1) Encode text, then free encoder
        text_embedder = get_text_embedder(
            text_embedder_conf,
            device=accelerator.device if not getattr(args, "text_encoder_cpu", False) else "cpu",
            quantized_qwen=getattr(args, "quantized_qwen", False),
            qwen_auto=getattr(args, "text_encoder_auto", False),
        )

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(text_embedder)
        del text_embedder
        clean_memory_on_device(accelerator.device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["text_embeds"] = te_outputs_1[p]["text_embeds"]
            prompt_dict_copy["pooled_embed"] = te_outputs_1[p]["pooled_embed"]
            prompt_dict_copy["null_text_embeds"] = te_outputs_1[p]["text_embeds"]
            prompt_dict_copy["null_pooled_embed"] = te_outputs_1[p]["pooled_embed"]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        # Training always uses cached latents, so defer VAE load until sampling time.
        self._vae_checkpoint_path = vae_path or self.task_conf.vae.checkpoint_path
        self.vae_scaling_factor = 1.0

        class _VaeStub:
            def requires_grad_(self, *_, **__):
                return self

            def eval(self):
                return self

            def to(self, *_, **__):
                return self

        # Return a stub to satisfy base trainer expectations; real VAE is loaded only inside sample_images.
        self.vae = _VaeStub()
        return self.vae

    def _load_dit_config(self, args: argparse.Namespace) -> dict:
        conf = self.task_conf.dit_params
        if args.override_dit:
            conf = SimpleNamespace(**json.loads(args.override_dit))
        conf_dict = conf.__dict__ if isinstance(conf, SimpleNamespace) else conf.__dict__
        # Respect global attention flags for Kandinsky5
        if getattr(args, "flash_attn", False):
            conf_dict["attention_engine"] = "flash_attention_2"
        elif getattr(args, "flash3", False):
            conf_dict["attention_engine"] = "flash_attention_3"
        elif getattr(args, "sage_attn", False):
            conf_dict["attention_engine"] = "sage"
        elif getattr(args, "xformers", False):
            conf_dict["attention_engine"] = "xformers"
        elif getattr(args, "sdpa", False):
            conf_dict["attention_engine"] = "sdpa"
        else:
            conf_dict.setdefault("attention_engine", "auto")
        return conf_dict

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        def _detect_fp8_checkpoint(path: str, disable_numpy_memmap: bool = False) -> bool:
            with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
                for key in f.keys():
                    if key.endswith(".scale_weight"):
                        return True
            return False

        def _load_state_dict_stream(
            path: str, device: str | torch.device = "cpu", dtype: torch.dtype | None = None, disable_numpy_memmap: bool = False
        ):
            dev = torch.device(device)
            sd: dict[str, torch.Tensor] = {}
            with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key, device=dev)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    sd[key] = tensor
            return sd

        self.dit_conf = self._load_dit_config(args)
        # Keep the base model dtype at standard precision even when fp8_base is requested,
        # to match the stable inference loader behavior (fp8 is applied via monkey patch/quantization later).
        if args.fp8_base and not args.fp8_scaled:
            dit_weight_dtype = None
        with init_empty_weights():
            model = get_dit(self.dit_conf)
            if dit_weight_dtype is not None:
                model.to(dit_weight_dtype)

        # fp8 weights must live on GPU when possible; adjust quantization device based on swap usage.
        use_fp8 = args.fp8_scaled or args.fp8_base
        blocks_to_swap = getattr(args, "blocks_to_swap", 0) or 0
        quant_device = accelerator.device if use_fp8 else loading_device

        ckpt_path = dit_path or self.task_conf.checkpoint_path
        logger.info(f"Loading DiT from {ckpt_path}")
        disable_memmap = getattr(args, "disable_numpy_memmap", False)
        is_fp8_ckpt = _detect_fp8_checkpoint(ckpt_path, disable_numpy_memmap=disable_memmap)

        # Load state dict
        state_dict = _load_state_dict_stream(ckpt_path, device="cpu", dtype=None, disable_numpy_memmap=disable_memmap)

        if is_fp8_ckpt:
            # fp8 checkpoint: use as-is, just apply monkey patch (like HunyuanVideo)
            logger.info("Checkpoint contains fp8 weights; using as-is.")
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=args.fp8_fast)
            dit_weight_dtype = None
            use_fp8 = False  # skip re-quantization below
        elif use_fp8:
            logger.info(f"Applying fp8 optimization (scaled={args.fp8_scaled}, base={args.fp8_base}) on {quant_device}")
            # If block swap is disabled, keep weights on GPU for speed; otherwise keep them on CPU to avoid OOM.
            if blocks_to_swap == 0:
                move_to_device = True
                fp8_quant_device = quant_device  # GPU quant/keep
            else:
                move_to_device = False
                # quantize on GPU even when block swap is on, but keep weights on CPU afterwards for swap
                fp8_quant_device = accelerator.device if quant_device == "cpu" else quant_device
            state_dict = model.fp8_optimization(state_dict, fp8_quant_device, move_to_device, use_scaled_mm=args.fp8_fast)
            dit_weight_dtype = None

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded DiT weights: {info}")
        # free CPU copy ASAP
        del state_dict

        model.attention = SimpleNamespace(**self.task_conf.attention.__dict__)

        model.dtype = next(model.parameters()).dtype  # align hv_train_network logging expectation
        model.device = loading_device  # align hv_train_network logging expectation

        return model

    def compile_transformer(self, args, transformer):
        transformer: DiffusionTransformer3D = transformer
        k5_nn.activate_compile(
            mode=args.compile_mode, backend=args.compile_backend, fullgraph=args.compile_fullgraph, dynamic=args.compile_dynamic
        )
        return transformer

    def scale_shift_latents(self, latents):
        # Latents were scaled during caching; avoid re-scaling during training.
        return latents

    def _load_vae_for_sampling(self, args: argparse.Namespace, device: torch.device):
        vae_conf = SimpleNamespace(name=self.task_conf.vae.name, checkpoint_path=self._vae_checkpoint_path)
        # Decode has been unstable in fp16 on some GPUs; prefer float32 for sampling.
        target_dtype = args.vae_dtype if device.type == "cuda" else torch.float32
        target_dtype = ensure_dtype_form(target_dtype, out_form="torch")
        disable_vae_workaround = getattr(args, "disable_vae_workaround", False)
        vae = build_vae(vae_conf, vae_dtype=target_dtype, enable_safety=not disable_vae_workaround)
        vae = vae.to(device=device, dtype=target_dtype)
        vae.eval()
        return vae

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: DiffusionTransformer3D,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        bsz = latents.shape[0]
        preds = []
        targets = []

        patch_size = self.dit_conf["patch_size"] if self.dit_conf else (1, 2, 2)

        for b in range(bsz):
            latent_b = latents[b].to(accelerator.device, dtype=network_dtype)
            noise_b = noise[b].to(accelerator.device, dtype=network_dtype)
            noisy_input_b = noisy_model_input[b].to(accelerator.device, dtype=network_dtype)
            # ensure the hidden state will require grad
            if args.gradient_checkpointing:
                noisy_input_b.requires_grad_(True)
            text_embed = batch["text_embeds"][b].to(accelerator.device, dtype=network_dtype)
            pooled_embed = batch["pooled_embed"][b].to(accelerator.device, dtype=network_dtype)

            # latents can be image (C, H, W) or video (C, F, H, W)
            if latent_b.dim() == 4:
                duration = latent_b.shape[-3]
                height, width = latent_b.shape[-2:]
                x = noisy_input_b.permute(1, 2, 3, 0)  # C, F, H, W -> F, H, W, C

                # append visual conditioning channels if model expects them (zeros by default)
                if transformer.visual_cond:
                    visual_cond = torch.zeros_like(x)  # [F,H,W,C]
                    visual_cond_mask = torch.zeros((*x.shape[:-1], 1), device=accelerator.device, dtype=network_dtype)  # [F,H,W,1]

                    cond_lat = None
                    if self._i2v_training:
                        assert "latents_image" in batch, (
                            "latents_image not found in batch; run kandinsky5_cache_latents to populate I2V caches"
                        )
                        # cached as C,F,H,W with F=2 (first+last universal)
                        cond_lat = batch["latents_image"][b].to(accelerator.device, dtype=network_dtype)
                        cond_lat = cond_lat.permute(1, 2, 3, 0)  # -> [2,H,W,C] (first, last)

                    if cond_lat is not None:
                        # Always condition the first frame
                        visual_cond[0] = cond_lat[0]
                        visual_cond_mask[0] = 1

                        # Optionally condition the last frame too (only if the video has >1 frame)
                        if self._i2v_mode == "first_last" and x.shape[0] > 1:
                            visual_cond[-1] = cond_lat[-1]
                            visual_cond_mask[-1] = 1

                    x = torch.cat([x, visual_cond, visual_cond_mask], dim=-1)
            else:
                duration = 1
                height, width = latent_b.shape[-2:]
                x = noisy_input_b.permute(1, 2, 0).unsqueeze(0)  # C, H, W -> 1, H, W, C
                if transformer.visual_cond:
                    visual_cond = torch.zeros_like(x)
                    visual_cond_mask = torch.zeros((*x.shape[:-1], 1), device=accelerator.device, dtype=network_dtype)
                    x = torch.cat([x, visual_cond, visual_cond_mask], dim=-1)

            sparse_params = self._build_sparse_params(x, x.device)

            visual_rope_pos = [
                torch.arange(duration, device=accelerator.device),
                torch.arange(height // patch_size[1], device=accelerator.device),
                torch.arange(width // patch_size[2], device=accelerator.device),
            ]
            text_rope_pos = torch.arange(text_embed.shape[0], device=accelerator.device)

            t_b = timesteps[b]
            if t_b.dim() > 0:
                t_b = t_b.flatten()[0]
            t_b = t_b.to(accelerator.device, dtype=network_dtype).unsqueeze(0)

            with accelerator.autocast():
                model_pred = transformer(
                    x,
                    text_embed,
                    pooled_embed,
                    t_b,
                    visual_rope_pos,
                    text_rope_pos,
                    scale_factor=tuple(self.task_conf.scale_factor),
                    sparse_params=sparse_params,
                    attention_mask=None,
                )

            # transformer outputs [duration, H, W, C]; align to [duration, C, H, W]
            model_pred = model_pred.permute(0, 3, 1, 2)
            target_d = noise_b - latent_b
            if target_d.dim() == 4:
                # C, F, H, W -> F, C, H, W to match duration
                target_d = target_d.permute(1, 0, 2, 3)
            else:
                target_d = target_d.unsqueeze(0)
            preds.append(model_pred)
            targets.append(target_d)

        model_pred = torch.stack(preds, dim=0)  # B, F, C, H, W
        target = torch.stack(targets, dim=0)  # B, F, C, H, W
        return model_pred, target

    # endregion model specific


def kandinsky5_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, required=True, help="Kandinsky5 task key (see configs.TASK_CONFIGS)")
    parser.add_argument("--override_dit", type=str, default=None, help="JSON dict to override DiT params")
    # fp8 and block swap flags are defined in setup_parser_common; reuse them to avoid conflicts
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder_qwen", type=str, default=None, help="Override Qwen text encoder path")
    parser.add_argument("--text_encoder_clip", type=str, default=None, help="Override CLIP text encoder path")
    parser.add_argument(
        "--i2v_mode",
        type=str,
        default="first",
        choices=["first", "first_last"],
        help="I2V conditioning mode: first frame only (default) or first+last frame.",
    )
    parser.add_argument(
        "--force_nabla_attention", action="store_true", help="Force nabla attention for training regardless of task default"
    )
    parser.add_argument("--nabla_P", type=float, default=0.9, help="CDF threshold P for nabla attention (default 0.9)")
    parser.add_argument("--nabla_wT", type=int, default=11, help="Temporal STA window for nabla attention (default 11)")
    parser.add_argument("--nabla_wH", type=int, default=3, help="Height STA window for nabla attention (default 3)")
    parser.add_argument("--nabla_wW", type=int, default=3, help="Width STA window for nabla attention (default 3)")
    parser.add_argument("--nabla_method", type=str, default="topcdf", help="Nabla map binarization method (default topcdf)")
    parser.add_argument(
        "--nabla_add_sta",
        dest="nabla_add_sta",
        action="store_true",
        default=True,
        help="Include STA prior when forcing nabla attention (default: True)",
    )
    parser.add_argument(
        "--no_nabla_add_sta", dest="nabla_add_sta", action="store_false", help="Disable STA prior when forcing nabla attention"
    )
    parser.add_argument("--quantized_qwen", action="store_true", help="Load Qwen text encoder in 4bit mode")
    parser.add_argument("--text_encoder_cpu", action="store_true", help="Run Qwen TE on CPU")
    parser.add_argument("--text_encoder_auto", action="store_true", help="Run Qwen with device_map=auto")

    return parser


def main():
    parser = setup_parser_common()
    parser = kandinsky5_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    if sum([args.text_encoder_cpu, args.quantized_qwen, args.text_encoder_auto]) > 1:
        raise ValueError(
            "Only one of '--quantized_qwen', '--text_encoder_cpu', '--text_encoder_auto' may be used at a time but received more than that!"
        )
    # defaults for fp8 flags (not defined in common parser)
    if not hasattr(args, "fp8_base"):
        args.fp8_base = False
    if not hasattr(args, "fp8_scaled"):
        args.fp8_scaled = False
    if not hasattr(args, "fp8_fast"):
        args.fp8_fast = False
    if not hasattr(args, "blocks_to_swap"):
        args.blocks_to_swap = 0
    if not hasattr(args, "dit_dtype"):
        args.dit_dtype = None
    # Avoid casting the entire DiT to float8 during sampling/training when fp8_base is used.
    # Setting fp8_scaled=True keeps the loader's fp8 quantization path but prevents the downstream float8 cast in hv_train_network.
    if args.fp8_base and not args.fp8_scaled:
        args.fp8_scaled = True

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Kandinsky5NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
