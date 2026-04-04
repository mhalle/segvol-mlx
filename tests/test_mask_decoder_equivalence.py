import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "SegVol"))
sys.path.insert(0, str(REPO_ROOT / "segvol-mlx" / "src"))

from segment_anything_volumetric.modeling.mask_decoder import MaskDecoder as PTMaskDecoder
from segment_anything_volumetric.modeling.transformer import TwoWayTransformer as PTTwoWayTransformer
from segvol_mlx.mask_decoder import MaskDecoder as MLXMaskDecoder
from segvol_mlx.weights import remap_segvol_key


def _load_decoder_weights(pt_dec: PTMaskDecoder, mlx_dec: MLXMaskDecoder) -> None:
    weight_pairs = []
    for key, value in pt_dec.state_dict().items():
        orig_key = f"mask_decoder.{key}"
        arr = value.detach().cpu().numpy()
        if arr.ndim == 5:
            if "output_upscaling.0.weight" in orig_key or "output_upscaling.3.weight" in orig_key:
                arr = arr.transpose(1, 2, 3, 4, 0)
            else:
                arr = arr.transpose(0, 2, 3, 4, 1)
        elif "output_upscaling.1." in orig_key and arr.ndim == 4:
            arr = arr.transpose(1, 2, 3, 0)
        new_key = remap_segvol_key(orig_key)[len("mask_decoder."):]
        weight_pairs.append((new_key, mx.array(arr)))
    mlx_dec.load_weights(weight_pairs, strict=True)


def test_mask_decoder_matches_pytorch():
    np.random.seed(1)
    torch.manual_seed(1)
    mx.random.seed(1)

    pt_dec = PTMaskDecoder(
        image_encoder_type="vit",
        transformer_dim=768,
        transformer=PTTwoWayTransformer(depth=2, embedding_dim=768, mlp_dim=2048, num_heads=8),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        image_size=np.array((32, 256, 256)),
        patch_size=np.array((4, 16, 16)),
    )
    pt_dec.eval()

    mlx_dec = MLXMaskDecoder(
        transformer_dim=768,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        image_size=(32, 256, 256),
        patch_size=(4, 16, 16),
    )
    _load_decoder_weights(pt_dec, mlx_dec)

    img_emb_np = np.random.randn(1, 768, 8, 16, 16).astype(np.float32)
    img_pe_np = np.random.randn(1, 768, 8, 16, 16).astype(np.float32)
    sparse_np = np.random.randn(1, 7, 768).astype(np.float32)
    dense_np = np.random.randn(1, 768, 8, 16, 16).astype(np.float32)
    text_np = np.random.randn(1, 768).astype(np.float32)

    with torch.no_grad():
        masks_pt, iou_pt = pt_dec.predict_masks(
            image_embeddings=torch.from_numpy(img_emb_np),
            text_embedding=torch.from_numpy(text_np),
            image_pe=torch.from_numpy(img_pe_np),
            sparse_prompt_embeddings=torch.from_numpy(sparse_np),
            dense_prompt_embeddings=torch.from_numpy(dense_np),
        )
    masks_pt = masks_pt.detach().cpu().numpy()
    iou_pt = iou_pt.detach().cpu().numpy()

    masks_mlx, iou_mlx = mlx_dec.predict_masks(
        mx.array(img_emb_np),
        mx.array(img_pe_np),
        mx.array(sparse_np),
        mx.array(dense_np),
        mx.array(text_np),
    )
    mx.eval(masks_mlx, iou_mlx)
    masks_mlx = np.array(masks_mlx)
    iou_mlx = np.array(iou_mlx)

    assert np.max(np.abs(masks_pt - masks_mlx)) < 1e-5
    assert np.max(np.abs(iou_pt - iou_mlx)) < 1e-5
