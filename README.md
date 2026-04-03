# segvol-mlx

MLX port of [SegVol](https://github.com/BAAI-DCAI/SegVol) (NeurIPS 2024) — text-prompted 3D medical image segmentation on Apple Silicon.

SegVol is the first volumetric medical segmentation model that accepts **text prompts** ("segment the liver"), **point clicks**, and **bounding boxes**. This port runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) with pretrained weights from [BAAI/SegVol](https://huggingface.co/BAAI/SegVol).

## Status

**Alpha** — working with real pretrained weights on real CT data.

| Component | Status |
|---|---|
| ViT encoder (12 layers, 768 dim) | Working, weights loaded |
| CLIP text encoder (native MLX) | Working |
| SAM-style mask decoder + text alignment | Working |
| Prompt encoder (point, box, text) | Working |
| 462/462 weights loaded (strict) | Clean |
| **Fits M2 16GB** | **Yes (~3GB)** |

## Quick start

```bash
git clone https://github.com/mhalle/segvol-mlx.git
cd segvol-mlx
uv sync
```

### Text-prompted segmentation

```python
from segvol_mlx.weights import download_and_load
from segvol_mlx.text_encoder import load_tokenizer
import mlx.core as mx

model = download_and_load()
tokenizer = load_tokenizer()

# Preprocess CT to (32, 256, 256), MinMax normalized
patch = ...  # your preprocessed CT slab

# Segment by organ name
masks = model.segment_by_text(
    mx.array(patch[None, :, :, :, None]),
    "liver",
    tokenizer,
)
mx.eval(masks)
segmentation = (masks > 0).astype(mx.int32)
```

### Point-prompted segmentation

```python
masks, iou = model(
    mx.array(patch[None, :, :, :, None]),
    points=(
        mx.array([[[16.0, 128.0, 128.0]]]),  # click location (D, H, W)
        mx.array([[1.0]]),                     # 1=positive, 0=negative
    ),
    multimask_output=False,
)
```

### Box + text (strongest mode)

```python
from segvol_mlx.text_encoder import SEGVOL_TEXT_TEMPLATE, tokenize_text

text = SEGVOL_TEXT_TEMPLATE.format("left lung upper lobe")
text_emb = model.text_encoder(tokenize_text(tokenizer, text))

masks, iou = model(
    mx.array(patch[None, :, :, :, None]),
    boxes=mx.array([[5.0, 50.0, 50.0, 25.0, 200.0, 200.0]]),  # d1,h1,w1,d2,h2,w2
    text_embedding=text_emb,
    multimask_output=False,
)
```

## Results on real CT (M2 16GB)

Single 32×256×256 slab, no sliding window or zoom-in-zoom-out:

| Organ | Point only | Text + box | Time |
|---|---|---|---|
| Left lung upper lobe | 0.86 | **0.94** | 0.43s |
| Spleen | 0.68 | — | 0.43s |
| Right lung upper lobe | 0.65 | — | 0.43s |
| Liver | 0.41 | — | 0.43s |

Text + box prompts produce the best results (0.94 Dice on lung). Zoom-in-zoom-out inference (not yet implemented) would improve all numbers further.

## Architecture

```
SegVol (~107M params)
├── image_encoder: ViT (86M, 12 layers, 768 dim, global attention)
│   ├── PatchEmbed3D: Conv3d(1→768, kernel=(4,16,16), stride=(4,16,16))
│   ├── Learned position embeddings (1, 2048, 768)
│   ├── 12× TransformerBlock (LayerNorm → MultiHeadAttn → MLP)
│   └── Final LayerNorm → reshape to (B, 768, 8, 16, 16)
├── text_encoder: CLIP (63M frozen + 0.4M alignment)
│   ├── CLIPTextModel (12 layers, 512 dim) — native MLX implementation
│   └── dim_align: Linear(512→768)
├── prompt_encoder: SAM-style
│   ├── PositionEmbeddingRandom (random Fourier features → 768 dim)
│   └── 4 point embeddings (positive, negative, box corner ×2)
└── mask_decoder: TwoWayTransformer + text alignment
    ├── TwoWayTransformer(depth=2, dim=768, heads=8, mlp=2048)
    ├── ConvTranspose3d upscaling (8×16×16 → 32×64×64)
    ├── 4× hypernetwork MLP → mask generation
    └── txt_align: Linear(768→96) for text-image similarity
```

## Preprocessing

SegVol expects 32-slice CT slabs at 256×256 in-plane:

```python
import numpy as np
from scipy.ndimage import zoom

# 1. MinMax normalize
ct_norm = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min())

# 2. Extract 32-slice slab centered on region of interest
slab = ct_norm[start:start+32, :, :]

# 3. Resize to (32, 256, 256)
scale = np.array([32, 256, 256]) / np.array(slab.shape)
slab_resized = zoom(slab, scale, order=1)
```

## MLX features used

- `mx.fast.scaled_dot_product_attention` — fused attention in both ViT encoder and SAM decoder
- `nn.Conv3d`, `nn.ConvTranspose3d` — 3D patch embedding and mask upscaling
- Native CLIP text encoder (no PyTorch dependency at runtime)

## Memory and performance

| | SegVol (32×256×256) | VISTA3D (128³) | SAM-Med3D (128³) |
|---|---|---|---|
| Memory | **~3GB** | ~22GB (OOM on 16GB) | ~5GB |
| Encoder time | **0.28s** | 0.43s (64³ only) | 0.39s |
| Total time | **0.43s** | 0.74s | 0.45s |
| Fits 16GB? | **Yes** | No | Yes |
| Text prompts | **Yes** | No | No |

## Related work

- [vista3d-mlx](https://github.com/mhalle/vista3d-mlx) — VISTA3D port (128³, needs 32GB, 133 auto classes)
- [sam-med3d-mlx](https://github.com/mhalle/sam-med3d-mlx) — SAM-Med3D port (verified exact, weak model)
- [nnunet-mlx](https://github.com/mhalle/nnunet-mlx) — nnU-Net / TotalSegmentator (production, 105 classes)
- [monai-mlx](https://github.com/mhalle/monai-mlx) — 6 MONAI architectures

## TODO

- Zoom-in-zoom-out inference pipeline
- Precomputed text embeddings for all 200+ organ classes
- Sliding window for full-volume coverage
- fp16 support

## License

Apache 2.0

## Citation

```bibtex
@article{du2024segvol,
  title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation},
  author={Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo},
  journal={NeurIPS},
  year={2024}
}
```
