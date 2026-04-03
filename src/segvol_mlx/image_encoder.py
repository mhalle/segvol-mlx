"""3D Vision Transformer image encoder for SegVol, ported to MLX.

Uses MONAI-style ViT with global self-attention (no window attention,
no relative position embeddings). Simpler than SAM-Med3D's encoder.
"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class PatchEmbed3D(nn.Module):
    """3D patch embedding via Conv3d."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 768,
                 patch_size: Tuple[int, int, int] = (4, 16, 16)):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.patch_size = patch_size

    def __call__(self, x: mx.array) -> mx.array:
        # Input: (B, D, H, W, C_in) channels-last
        x = self.proj(x)  # (B, D', H', W', embed_dim)
        B = x.shape[0]
        embed_dim = x.shape[-1]
        # Flatten spatial to tokens: (B, num_tokens, embed_dim)
        return x.reshape(B, -1, embed_dim)


class MLPBlock(nn.Module):
    """Transformer MLP: Linear → GELU → Linear."""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(nn.gelu(self.lin1(x)))


class TransformerBlock(nn.Module):
    """Standard transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_dim: int = 3072):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head self-attention with combined QKV projection."""

    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class ViTEncoder(nn.Module):
    """MONAI-style 3D ViT encoder for SegVol.

    Input: (B, D, H, W, 1) channels-last CT volume
    Output: (B, num_tokens, 768) token embeddings

    With default config (32×256×256 input, 4×16×16 patches):
    num_tokens = 8×16×16 = 2048
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: Tuple[int, int, int] = (32, 256, 256),
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbed3D(in_channels, embed_dim, patch_size)

        grid_size = tuple(s // p for s, p in zip(img_size, patch_size))
        num_tokens = grid_size[0] * grid_size[1] * grid_size[2]
        self.grid_size = grid_size
        self.num_tokens = num_tokens

        # Learned position embeddings
        self.position_embeddings = mx.zeros((1, num_tokens, embed_dim))

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ]

        # Final norm (MONAI ViT has this)
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, D, H, W, 1) channels-last input
        Returns:
            (B, num_tokens, embed_dim) token embeddings
        """
        x = self.patch_embedding(x)  # (B, num_tokens, embed_dim)
        x = x + self.position_embeddings

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
