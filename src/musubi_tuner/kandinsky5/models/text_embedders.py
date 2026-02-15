# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, CLIPTextModel, CLIPTokenizer, BitsAndBytesConfig

from .utils import freeze
import torchvision.transforms.functional as F


class ClipTextEmbedder:
    def __init__(self, conf, device):
        self.model = CLIPTextModel.from_pretrained(conf.checkpoint_path).to(device)
        self.model = freeze(self.model)
        self.tokenizer = CLIPTokenizer.from_pretrained(conf.checkpoint_path)
        self.max_length = conf.max_length

    def __call__(self, texts):
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            pooled_embed = self.model(**inputs)["pooler_output"]
        return pooled_embed


class Qwen2_5_VLTextEmbedder:
    INSTRUCTION_I2V_EXPAND = """You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
    Here are some example descriptions from the dataset that the model was trained:
    1. "Create a video showing a nighttime urban driving scene from inside a car. The driver is focused on the road ahead, with the city lights visible through the windshield. The GPS device on the dashboard continues to display navigation information. The camera remains steady, capturing the interior of the car and the changing street view outside as the vehicle moves forward. The background shifts slightly to show different parts of the cityscape, including illuminated buildings and street signs."
    2. "Create a video where the character, dressed in historical attire, is seen holding an umbrella with a logo. The character should move closer to the camera while maintaining a steady pace, keeping the umbrella raised. The background remains consistent with a foggy, outdoor setting, but the focus shifts more towards the character as they approach. The lighting should emphasize the details of the costume and the umbrella, enhancing the dramatic effect."
    3. "Darken the scene while keeping the characters and setting unchanged, emphasizing a serious atmosphere."
    IImportantly! These are just examples from a large training dataset of 20 mln videos.
    Rewrite Prompt: "{prompt}" to get high-quality image to video generation from this image. Pay main attention to information about changes of objects.
    Make prompt dynamic. Answer only with expanded prompt."""
    PROMPT_TEMPLATE = {
        "template": {
            "video": (
                "<|im_start|>system\nYou are a prompt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
            "image2video": (
                "<|im_start|>system\nYou are a prompt engineer. Your task is to create a highly detailed and effective video description based on a provided input image.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe main characters actions.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
            "image": (
                "<|im_start|>system\nYou are a prompt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
            "image_edit": (
                "<|im_start|>system\nYou are a prompt engineer. Based on the provided source image (first image) and target image (second image), create an interesting text prompt that can be used together with the source image to create the target image:<|im_end|>",
                "<|im_start|>user\n{}",
            ),
        },
        "crop_start": {"video": 129, "image": 41, "image_edit": 55, "image2video": 132},  # Deprecated, dynamic now
    }

    def __init__(self, conf, device, quantized_qwen=False):
        quantization_config = None
        if quantized_qwen:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            conf.checkpoint_path, torch_dtype=torch.bfloat16, device_map=device, quantization_config=quantization_config
        )
        self.model = freeze(self.model)

        self.processor = AutoProcessor.from_pretrained(conf.checkpoint_path, use_fast=True)
        self.max_length = conf.max_length

    def __call__(self, texts, images=None, type_of_content="video", use_system=True):
        if use_system:
            prompt_template = "\n".join(self.PROMPT_TEMPLATE["template"][type_of_content])

            # Split into prefix/suffix around the user slot
            if "{}" not in prompt_template:
                raise ValueError(f"Prompt template for '{type_of_content}' is missing '{{}}' slot.")
            prefix, suffix = prompt_template.split("{}", 1)

            # Compute crop_start dynamically from tokenized prefix
            # Prefer tokenizer directly to avoid any processor padding/truncation behavior
            tok = getattr(self.processor, "tokenizer", None)
            if tok is None:
                # Fallback: processor can usually tokenize text-only
                prefix_ids = self.processor(
                    text=[prefix],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )["input_ids"][0]
                crop_start = int(prefix_ids.shape[0])
            else:
                prefix_ids = tok(
                    prefix,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
                )["input_ids"][0]
                crop_start = int(prefix_ids.shape[0])

            # Build full texts using the same prefix/suffix
            full_texts = [prefix + t + suffix for t in texts]
        else:
            full_texts = texts
            crop_start = 0

        if type_of_content == "image_edit":
            # Normalize images to a list aligned with full_texts
            if images is not None:
                if isinstance(images, torch.Tensor):
                    # [B,C,H,W] or [C,H,W]
                    if images.dim() == 3:
                        images = images.unsqueeze(0)
                    images = [images[i] for i in range(images.shape[0])]
                elif not isinstance(images, (list, tuple)):
                    images = [images]

                # If one image provided for many prompts, repeat it
                if len(images) == 1 and len(full_texts) > 1:
                    images = images * len(full_texts)

                for i in range(len(full_texts)):
                    full_texts[i] = full_texts[i] + "<|vision_start|><|image_pad|><|vision_end|><|im_end|>"

                images = [F.resize(im, (im.shape[-2] // 2, im.shape[-1] // 2)) for im in images]

            inputs = self.processor(
                text=full_texts,
                images=images,
                truncation=True,
                return_tensors="pt",
                padding=True,
                max_length=None,  # safer when images present
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                embeds = out["hidden_states"][-1][:, crop_start:]
        else:
            max_length = self.max_length + crop_start
            inputs = self.processor(
                text=full_texts,
                images=None,
                videos=None,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                padding="max_length",
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                    output_hidden_states=True,
                )
                embeds = out["hidden_states"][-1][:, crop_start:]

        attention_mask = inputs["attention_mask"][:, crop_start:]

        # Pack to [T, D] and build cu_seqlens
        bsz = attention_mask.shape[0]
        seqlens = attention_mask.sum(1).to(torch.int32)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=seqlens.device), torch.cumsum(seqlens, 0)])
        embeds = torch.cat([embeds[i][attention_mask[i].bool()] for i in range(bsz)], dim=0)

        return embeds, cu_seqlens

    def expand_text_prompt(self, prompt, image, device="cuda"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": self.INSTRUCTION_I2V_EXPAND.format(prompt=prompt),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]


class Kandinsky5TextEmbedder:
    def __init__(self, conf, device="cpu", quantized_qwen=False, qwen_auto=False):
        self.embedder = Qwen2_5_VLTextEmbedder(conf.qwen, device if not qwen_auto else "auto", quantized_qwen)
        self.clip_embedder = ClipTextEmbedder(conf.clip, device)
        self.conf = conf

    def encode(self, texts, images=None, type_of_content="image", use_system=True):
        text_embeds, cu_seqlens = self.embedder(texts, images=images, type_of_content=type_of_content, use_system=use_system)
        pooled_embed = self.clip_embedder(texts)
        return {"text_embeds": text_embeds, "pooled_embed": pooled_embed}, cu_seqlens

    def to(self, device):
        self.embedder.model = self.embedder.model.to(device)
        self.clip_embedder.model = self.clip_embedder.model.to(device)
        return self


def get_text_embedder(conf, device="cpu", quantized_qwen=False, qwen_auto=False):
    return Kandinsky5TextEmbedder(conf, device, quantized_qwen, qwen_auto)
