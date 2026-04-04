"""SegVol weight loading for MLX."""

import re
from typing import Dict

import mlx.core as mx
import numpy as np


def remap_segvol_key(key: str) -> str:
    """Remap PyTorch SegVol checkpoint key to MLX model key."""
    k = key
    if k.startswith("model."):
        k = k[len("model."):]

    # --- Image encoder (MONAI ViT) ---
    # MONAI patch embedding: patch_embedding.patch_embeddings.1 -> patch_embedding.proj (Conv3d)
    # The Linear weight gets reshaped to Conv3d during loading, key just maps
    k = k.replace("image_encoder.patch_embedding.patch_embeddings.1.",
                   "image_encoder.patch_embedding.proj.")
    # MONAI position_embeddings are inside patch_embedding
    k = k.replace("image_encoder.patch_embedding.position_embeddings",
                   "image_encoder.position_embeddings")

    # MLP: linear1/2 -> lin1/2 (applies globally)
    k = k.replace(".mlp.linear1.", ".mlp.lin1.")
    k = k.replace(".mlp.linear2.", ".mlp.lin2.")

    # --- CLIP text encoder ---
    k = k.replace("text_encoder.clip_text_model.text_model.embeddings.token_embedding.",
                   "text_encoder.clip_text_model.token_embedding.")
    k = k.replace("text_encoder.clip_text_model.text_model.embeddings.position_embedding.",
                   "text_encoder.clip_text_model.position_embedding.")
    k = k.replace("text_encoder.clip_text_model.text_model.encoder.layers.",
                   "text_encoder.clip_text_model.encoder_layers.")
    k = k.replace("text_encoder.clip_text_model.text_model.final_layer_norm.",
                   "text_encoder.clip_text_model.final_layer_norm.")
    # CLIP MLP: fc1/fc2 -> mlp_fc1/mlp_fc2
    k = k.replace(".mlp.fc1.", ".mlp_fc1.")
    k = k.replace(".mlp.fc2.", ".mlp_fc2.")

    # --- Mask decoder ---
    # Output upscaling: Sequential indices -> named attributes
    k = k.replace("mask_decoder.output_upscaling.0.", "mask_decoder.output_upscaling_conv1.")
    k = k.replace("mask_decoder.output_upscaling.1.", "mask_decoder.output_upscaling_norm1.")
    k = k.replace("mask_decoder.output_upscaling.3.", "mask_decoder.output_upscaling_conv2.")

    return k


# Keys to skip
_SKIP_KEYS = {
    "model.text_encoder.clip_text_model.text_model.embeddings.position_ids",
}


def is_conv_transpose_weight(key: str) -> bool:
    return "output_upscaling.0.weight" in key or "output_upscaling.3.weight" in key


def _convert_patch_embed_weight(weight: np.ndarray, patch_size=(4, 16, 16)) -> np.ndarray:
    """Convert MONAI Linear patch embedding weight to Conv3d format.

    MONAI: Linear(1024, 768) with weight shape (768, 1024)
    Conv3d equivalent: (768, 1, 4, 16, 16) -> MLX channels-last: (768, 4, 16, 16, 1)
    """
    # (768, 1024) -> (768, 1, 4, 16, 16) -> transpose to (768, 4, 16, 16, 1)
    out_ch = weight.shape[0]
    w = weight.reshape(out_ch, 1, *patch_size)  # (768, 1, 4, 16, 16) PyTorch NCDHW
    w = w.transpose(0, 2, 3, 4, 1)  # (768, 4, 16, 16, 1) MLX NDHWC
    return w


def load_segvol(checkpoint_path: str, dtype: str = "float32"):
    """Load SegVol from a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to pytorch_model.bin
        dtype: "float32" or "float16"
    """
    import torch
    from .segvol import SegVol
    from .image_encoder import ViTEncoder
    from .prompt_encoder import PromptEncoder
    from .mask_decoder import MaskDecoder
    from .text_encoder import TextEncoder

    enc = ViTEncoder(
        in_channels=1, img_size=(32, 256, 256), patch_size=(4, 16, 16),
        embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
    )
    pe = PromptEncoder(
        embed_dim=768, image_embedding_size=(8, 16, 16),
        input_image_size=(32, 256, 256), mask_in_chans=16,
    )
    dec = MaskDecoder(
        transformer_dim=768, num_multimask_outputs=3,
        iou_head_depth=3, iou_head_hidden_dim=256,
        image_size=(32, 256, 256), patch_size=(4, 16, 16),
    )
    te = TextEncoder()
    model = SegVol(enc, pe, dec, te)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_dtype = mx.float16 if dtype == "float16" else mx.float32

    mlx_weights = {}
    skipped = []
    for pt_key, tensor in state_dict.items():
        if pt_key in _SKIP_KEYS:
            continue

        arr = tensor.cpu().numpy()
        mlx_key = remap_segvol_key(pt_key)

        # Special handling: MONAI Linear patch embedding -> Conv3d
        if "patch_embedding.proj.weight" in mlx_key and arr.ndim == 2:
            arr = _convert_patch_embed_weight(arr, patch_size=(4, 16, 16))
        # Conv3d 5D weights
        elif arr.ndim == 5:
            if is_conv_transpose_weight(pt_key):
                arr = arr.transpose(1, 2, 3, 4, 0)
            else:
                arr = arr.transpose(0, 2, 3, 4, 1)
        # Spatial LayerNorm affine weights: (C, D, H, W) -> (D, H, W, C)
        elif "output_upscaling_norm1" in mlx_key and arr.ndim == 4:
            arr = arr.transpose(1, 2, 3, 0)
        # Conv2d 4D weights (mask_downscaling)
        elif arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # Skip mask_downscaling (Conv2d — not implemented, not needed for text/point/box prompts)
        if "mask_downscaling" in mlx_key:
            skipped.append(mlx_key)
            continue

        mlx_weights[mlx_key] = mx.array(arr.astype(np.float32)).astype(target_dtype)

    try:
        model.load_weights(list(mlx_weights.items()), strict=True)
        print(f"Loaded {len(mlx_weights)} weights (strict)")
    except ValueError as e:
        print(f"Strict failed: {e}")
        model.load_weights(list(mlx_weights.items()), strict=False)
        print(f"Loaded {len(mlx_weights)} weights (non-strict)")

    if skipped:
        print(f"Skipped {len(skipped)} mask_downscaling weights (Conv2d, not needed for text/point/box)")

    return model


def download_and_load(dtype: str = "float32"):
    """Download SegVol from HuggingFace and load."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("BAAI/SegVol", "pytorch_model.bin")
    return load_segvol(path, dtype=dtype)
