# based on https://github.com/Stability-AI/ModelSpec
import datetime
import hashlib
from io import BytesIO
import os
from typing import List, Optional, Tuple, Union
import safetensors
from blissful_tuner.blissful_logger import BlissfulLogger

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_WAN,
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_FLUX_KONTEXT,
)

logger = BlissfulLogger(__name__, "green")


r"""
# Metadata Example
metadata = {
    # === Must ===
    "modelspec.sai_model_spec": "1.0.0", # Required version ID for the spec
    "modelspec.architecture": "stable-diffusion-xl-v1-base", # Architecture, reference the ID of the original model of the arch to match the ID
    "modelspec.implementation": "sgm",
    "modelspec.title": "Example Model Version 1.0", # Clean, human-readable title. May use your own phrasing/language/etc
    # === Should ===
    "modelspec.author": "Example Corp", # Your name or company name
    "modelspec.description": "This is my example model to show you how to do it!", # Describe the model in your own words/language/etc. Focus on what users need to know
    "modelspec.date": "2023-07-20", # ISO-8601 compliant date of when the model was created
    # === Can ===
    "modelspec.license": "ExampleLicense-1.0", # eg CreativeML Open RAIL, etc.
    "modelspec.usage_hint": "Use keyword 'example'" # In your own language, very short hints about how the user should use the model
}
"""

BASE_METADATA = {
    # === Must ===
    "modelspec.sai_model_spec": "1.0.0",  # Required version ID for the spec
    "modelspec.architecture": None,
    "modelspec.implementation": None,
    "modelspec.title": None,
    "modelspec.resolution": None,
    # === Should ===
    "modelspec.description": None,
    "modelspec.author": None,
    "modelspec.date": None,
    # === Can ===
    "modelspec.license": None,
    "modelspec.tags": None,
    "modelspec.merged_from": None,
    "modelspec.prediction_type": None,
    "modelspec.timestep_range": None,
    "modelspec.encoder_layer": None,
}

# 別に使うやつだけ定義
MODELSPEC_TITLE = "modelspec.title"

ARCH_HUNYUAN_VIDEO = "hunyuan-video"

# Official Wan2.1 weights does not have sai_model_spec, so we use this as an architecture name
ARCH_WAN = "wan2.1"

ARCH_FRAMEPACK = "framepack"

ADAPTER_LORA = "lora"

IMPL_HUNYUAN_VIDEO = "https://github.com/Tencent/HunyuanVideo"
IMPL_WAN = "https://github.com/Wan-Video/Wan2.1"
IMPL_FRAMEPACK = "https://github.com/lllyasviel/FramePack"
IMPL_FLUX_KONTEXT = "https://github.com/black-forest-labs/flux"

PRED_TYPE_EPSILON = "epsilon"
# PRED_TYPE_V = "v"


def load_bytes_in_safetensors(tensors):
    bytes = safetensors.torch.save(tensors)
    b = BytesIO(bytes)

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)

    return b.read()


def precalculate_safetensors_hashes(state_dict):
    # calculate each tensor one by one to reduce memory usage
    hash_sha256 = hashlib.sha256()
    for tensor in state_dict.values():
        single_tensor_sd = {"tensor": tensor}
        bytes_for_tensor = load_bytes_in_safetensors(single_tensor_sd)
        hash_sha256.update(bytes_for_tensor)

    return f"0x{hash_sha256.hexdigest()}"


def update_hash_sha256(metadata: dict, state_dict: dict):
    raise NotImplementedError


def build_metadata(
    state_dict: Optional[dict],
    architecture: str,
    timestamp: float,
    title: Optional[str] = None,
    reso: Optional[Union[int, Tuple[int, int]]] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[str] = None,
    merged_from: Optional[str] = None,
    timesteps: Optional[Tuple[int, int]] = None,
    is_lora: bool = True,
):
    metadata = {}
    metadata.update(BASE_METADATA)

    # TODO implement if we can calculate hash without loading all tensors
    # if state_dict is not None:
    # hash = precalculate_safetensors_hashes(state_dict)
    # metadata["modelspec.hash_sha256"] = hash

    # arch = ARCH_HUNYUAN_VIDEO
    if architecture == ARCHITECTURE_HUNYUAN_VIDEO:
        arch = ARCH_HUNYUAN_VIDEO
        impl = IMPL_HUNYUAN_VIDEO
    elif architecture == ARCHITECTURE_WAN:
        arch = ARCH_WAN
        impl = IMPL_WAN
    elif architecture == ARCHITECTURE_FRAMEPACK:
        arch = ARCH_FRAMEPACK
        impl = IMPL_FRAMEPACK
    elif architecture == ARCHITECTURE_FLUX_KONTEXT:
        arch = ARCHITECTURE_FLUX_KONTEXT
        impl = IMPL_FLUX_KONTEXT
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if is_lora:
        arch += f"/{ADAPTER_LORA}"
    metadata["modelspec.architecture"] = arch

    metadata["modelspec.implementation"] = impl

    if title is None:
        title = "LoRA" if is_lora else "Hunyuan-Video"
        title += f"@{timestamp}"
    metadata[MODELSPEC_TITLE] = title

    if author is not None:
        metadata["modelspec.author"] = author
    else:
        del metadata["modelspec.author"]

    if description is not None:
        metadata["modelspec.description"] = description
    else:
        del metadata["modelspec.description"]

    if merged_from is not None:
        metadata["modelspec.merged_from"] = merged_from
    else:
        del metadata["modelspec.merged_from"]

    if license is not None:
        metadata["modelspec.license"] = license
    else:
        del metadata["modelspec.license"]

    if tags is not None:
        metadata["modelspec.tags"] = tags
    else:
        del metadata["modelspec.tags"]

    # remove microsecond from time
    int_ts = int(timestamp)

    # time to iso-8601 compliant date
    date = datetime.datetime.fromtimestamp(int_ts).isoformat()
    metadata["modelspec.date"] = date

    if reso is not None:
        # comma separated to tuple
        if isinstance(reso, str):
            reso = tuple(map(int, reso.split(",")))
        if len(reso) == 1:
            reso = (reso[0], reso[0])
    else:
        # resolution is defined in dataset, so use default
        reso = (1280, 720)
    if isinstance(reso, int):
        reso = (reso, reso)

    metadata["modelspec.resolution"] = f"{reso[0]}x{reso[1]}"

    # metadata["modelspec.prediction_type"] = PRED_TYPE_EPSILON
    del metadata["modelspec.prediction_type"]

    if timesteps is not None:
        if isinstance(timesteps, str) or isinstance(timesteps, int):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        metadata["modelspec.timestep_range"] = f"{timesteps[0]},{timesteps[1]}"
    else:
        del metadata["modelspec.timestep_range"]

    # if clip_skip is not None:
    #     metadata["modelspec.encoder_layer"] = f"{clip_skip}"
    # else:
    del metadata["modelspec.encoder_layer"]

    # # assert all values are filled
    # assert all([v is not None for v in metadata.values()]), metadata
    if not all([v is not None for v in metadata.values()]):
        logger.error(f"Internal error: some metadata values are None: {metadata}")

    return metadata


# region utils


def get_title(metadata: dict) -> Optional[str]:
    return metadata.get(MODELSPEC_TITLE, None)


def load_metadata_from_safetensors(model: str) -> dict:
    if not model.endswith(".safetensors"):
        return {}

    with safetensors.safe_open(model, framework="pt") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def build_merged_from(models: List[str]) -> str:
    def get_title(model: str):
        metadata = load_metadata_from_safetensors(model)
        title = metadata.get(MODELSPEC_TITLE, None)
        if title is None:
            title = os.path.splitext(os.path.basename(model))[0]  # use filename
        return title

    titles = [get_title(model) for model in models]
    return ", ".join(titles)


# endregion


r"""
if __name__ == "__main__":
    import argparse
    import torch
    from safetensors.torch import load_file
    from library import train_util

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading {args.ckpt}")
    state_dict = load_file(args.ckpt)

    print(f"Calculating metadata")
    metadata = get(state_dict, False, False, False, False, "sgm", False, False, "title", "date", 256, 1000, 0)
    print(metadata)
    del state_dict

    # by reference implementation
    with open(args.ckpt, mode="rb") as file_data:
        file_hash = hashlib.sha256()
        head_len = struct.unpack("Q", file_data.read(8))  # int64 header length prefix
        header = json.loads(file_data.read(head_len[0]))  # header itself, json string
        content = (
            file_data.read()
        )  # All other content is tightly packed tensors. Copy to RAM for simplicity, but you can avoid this read with a more careful FS-dependent impl.
        file_hash.update(content)
        # ===== Update the hash for modelspec =====
        by_ref = f"0x{file_hash.hexdigest()}"
    print(by_ref)
    print("is same?", by_ref == metadata["modelspec.hash_sha256"])

"""
