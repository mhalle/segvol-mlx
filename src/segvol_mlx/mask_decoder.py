"""Mask decoder for SegVol, ported to MLX.

TwoWayTransformer + ConvTranspose3d upscaling + text alignment.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU (except last layer)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = mx.sigmoid(x)
        return x


class DecoderAttention(nn.Module):
    """Multi-head attention with optional embedding downsampling."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        B = q.shape[0]
        q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.internal_dim)
        return self.out_proj(out)


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(nn.relu(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = DecoderAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int = 2, embedding_dim: int = 768, num_heads: int = 8,
                 mlp_dim: int = 2048, attention_downsample_rate: int = 2):
        super().__init__()
        self.layers = [
            TwoWayAttentionBlock(
                embedding_dim, num_heads, mlp_dim, attention_downsample_rate,
                skip_first_layer_pe=(i == 0))
            for i in range(depth)
        ]
        self.final_attn_token_to_image = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def __call__(self, image_embedding, image_pe, point_embedding):
        # image_embedding: (B, C, D, H, W) channels-first
        B, C = image_embedding.shape[0], image_embedding.shape[1]
        spatial = image_embedding.shape[2:]
        # Flatten: (B, C, D*H*W) → (B, D*H*W, C)
        image_flat = image_embedding.reshape(B, C, -1).transpose(0, 2, 1)
        image_pe_flat = image_pe.reshape(image_pe.shape[0], image_pe.shape[1], -1).transpose(0, 2, 1)
        # Broadcast PE if needed
        if image_pe_flat.shape[0] != B:
            image_pe_flat = mx.broadcast_to(image_pe_flat, (B,) + image_pe_flat.shape[1:])

        queries = point_embedding
        keys = image_flat
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe_flat)
        q = queries + point_embedding
        k = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class MaskDecoder(nn.Module):
    """Mask decoder with text alignment for SegVol."""

    def __init__(self, transformer_dim: int = 768, num_multimask_outputs: int = 3,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256,
                 image_size: Tuple[int, int, int] = (32, 256, 256),
                 patch_size: Tuple[int, int, int] = (4, 16, 16)):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_multimask_outputs + 1

        self.transformer = TwoWayTransformer(
            depth=2, embedding_dim=transformer_dim, mlp_dim=2048, num_heads=8)

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Output upscaling: (D/p, H/p, W/p) → 2× → 2× = (D/p*4, H/p*4, W/p*4)
        # With ViT patch (4,16,16) on (32,256,256): (8,16,16) → (16,32,32) → (32,64,64)
        self.output_upscaling_conv1 = nn.ConvTranspose3d(
            transformer_dim, transformer_dim // 4, kernel_size=2, stride=2)
        self.output_upscaling_norm1 = nn.LayerNorm(transformer_dim // 4)
        self.output_upscaling_conv2 = nn.ConvTranspose3d(
            transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2)

        out_dim = transformer_dim // 8  # 96

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, out_dim, 3)
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

        # Text-image alignment
        self.txt_align_upscaled_embedding = nn.Linear(768, out_dim)

    def __call__(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                 dense_prompt_embeddings, multimask_output: bool,
                 text_embedding: Optional[mx.array] = None):
        masks, iou_pred = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, text_embedding)

        if multimask_output:
            masks = masks[:, 1:, :, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            masks = masks[:, 0:1, :, :, :]
            iou_pred = iou_pred[:, 0:1]
        return masks, iou_pred

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings, text_embedding=None):
        # Concatenate output tokens
        iou_w = self.iou_token(mx.zeros((1,), dtype=mx.int32))
        mask_w = self.mask_tokens(mx.arange(self.num_mask_tokens))
        output_tokens = mx.concatenate([iou_w, mask_w], axis=0)
        output_tokens = mx.broadcast_to(
            output_tokens[None], (sparse_prompt_embeddings.shape[0],) + output_tokens.shape)
        tokens = mx.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c = src.shape[0], src.shape[1]
        spatial = src.shape[2:]  # (D, H, W)

        # Run transformer
        hs, src_out = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale: reshape back to spatial, channels-last for Conv3d
        src_spatial = src_out.reshape(b, *spatial, c)  # (B, D, H, W, C) channels-last

        x = self.output_upscaling_conv1(src_spatial)
        x = self.output_upscaling_norm1(x)
        x = nn.gelu(x)
        x = self.output_upscaling_conv2(x)
        x = nn.gelu(x)
        # x: (B, D', H', W', out_dim) channels-last

        up_spatial = x.shape[1:4]
        out_dim = x.shape[-1]
        x_flat = x.reshape(b, -1, out_dim)  # (B, D'*H'*W', out_dim)

        # Hypernetwork mask generation
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = mx.stack(hyper_in_list, axis=1)  # (B, num_masks, out_dim)

        masks = hyper_in @ x_flat.transpose(0, 2, 1)  # (B, num_masks, D'*H'*W')
        masks = masks.reshape(b, self.num_mask_tokens, *up_spatial)

        # Text-image alignment
        if text_embedding is not None:
            text_down = self.txt_align_upscaled_embedding(text_embedding)[:, None, :]  # (B, 1, out_dim)
            sim = text_down @ x_flat.transpose(0, 2, 1)  # (B, 1, D'*H'*W')
            sim = sim.reshape(b, 1, *up_spatial)
            sim = mx.broadcast_to(sim, masks.shape)
            masks = masks + sim

        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred
