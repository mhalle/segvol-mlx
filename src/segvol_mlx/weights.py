"""SegVol weight loading for MLX."""

import re
from typing import Dict

import mlx.core as mx
import numpy as np


def remap_segvol_key(key: str) -> str:
    """Remap PyTorch SegVol checkpoint key to MLX model key."""
    k = key
    # Strip 'model.' prefix
    if k.startswith("model."):
        k = k[len("model."):]

    # --- Image encoder (MONAI ViT) ---
    # MONAI ViT uses patch_embedding.* for the conv
    k = k.replace("image_encoder.patch_embedding.", "image_encoder.patch_embedding.proj.")
    # position_embeddings stays the same
    # blocks.N.attn.qkv stays (but MONAI uses combined qkv, not separate q/k/v)
    # blocks.N.attn.out_proj stays
    # blocks.N.mlp.linear1 -> blocks.N.mlp.lin1
    k = k.replace(".mlp.linear1.", ".mlp.lin1.")
    k = k.replace(".mlp.linear2.", ".mlp.lin2.")

    # --- CLIP text encoder ---
    # model.text_encoder.clip_text_model.text_model.embeddings.token_embedding
    #   → text_encoder.clip_text_model.token_embedding
    k = k.replace("text_encoder.clip_text_model.text_model.embeddings.token_embedding.",
                   "text_encoder.clip_text_model.token_embedding.")
    k = k.replace("text_encoder.clip_text_model.text_model.embeddings.position_embedding.",
                   "text_encoder.clip_text_model.position_embedding.")
    # Encoder layers
    k = k.replace("text_encoder.clip_text_model.text_model.encoder.layers.",
                   "text_encoder.clip_text_model.encoder_layers.")
    # Self-attention projections
    k = k.replace(".self_attn.q_proj.", ".self_attn.q_proj.")
    k = k.replace(".self_attn.k_proj.", ".self_attn.k_proj.")
    k = k.replace(".self_attn.v_proj.", ".self_attn.v_proj.")
    k = k.replace(".self_attn.out_proj.", ".self_attn.out_proj.")
    # MLP
    k = k.replace(".mlp.fc1.", ".mlp_fc1.")
    k = k.replace(".mlp.fc2.", ".mlp_fc2.")
    # Layer norms
    k = k.replace(".layer_norm1.", ".layer_norm1.")
    k = k.replace(".layer_norm2.", ".layer_norm2.")
    # Final layer norm
    k = k.replace("text_encoder.clip_text_model.text_model.final_layer_norm.",
                   "text_encoder.clip_text_model.final_layer_norm.")

    # --- Mask decoder ---
    # Transformer MLP naming: linear1/2 -> lin1/2
    # (already handled above for image_encoder, but also applies to decoder)
    # mask_decoder.transformer.layers.N.mlp.linear1 -> .mlp.lin1
    # (already handled by the global replace above)

    # Output upscaling: Sequential indices -> named attributes
    k = k.replace("mask_decoder.output_upscaling.0.", "mask_decoder.output_upscaling_conv1.")
    k = k.replace("mask_decoder.output_upscaling.1.", "mask_decoder.output_upscaling_norm1.")
    k = k.replace("mask_decoder.output_upscaling.3.", "mask_decoder.output_upscaling_conv2.")

    # --- Prompt encoder ---
    # pe_layer stays, mask_downscaling stays, point_embeddings stays

    return k


# Keys to skip (not model parameters)
_SKIP_KEYS = {
    "model.text_encoder.clip_text_model.text_model.embeddings.position_ids",
}


def is_conv_transpose_weight(key: str) -> bool:
    """Check if a key is a ConvTranspose3d weight."""
    return "output_upscaling.0.weight" in key


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

    # Build model
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

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_dtype = mx.float16 if dtype == "float16" else mx.float32

    mlx_weights = {}
    skipped = []
    for pt_key, tensor in state_dict.items():
        if pt_key in _SKIP_KEYS:
            continue

        arr = tensor.cpu().numpy()
        mlx_key = remap_segvol_key(pt_key)

        # Transpose conv weights
        if arr.ndim == 5:
            if is_conv_transpose_weight(pt_key):
                arr = arr.transpose(1, 2, 3, 4, 0)
            else:
                arr = arr.transpose(0, 2, 3, 4, 1)

        mlx_weights[mlx_key] = mx.array(arr.astype(np.float32)).astype(target_dtype)

    try:
        model.load_weights(list(mlx_weights.items()), strict=True)
        print(f"Loaded {len(mlx_weights)} weights (strict)")
    except ValueError as e:
        print(f"Strict failed: {e}")
        model.load_weights(list(mlx_weights.items()), strict=False)
        print(f"Loaded {len(mlx_weights)} weights (non-strict)")

    return model


def download_and_load(dtype: str = "float32"):
    """Download SegVol from HuggingFace and load."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("BAAI/SegVol", "pytorch_model.bin")
    return load_segvol(path, dtype=dtype)
