"""Prompt encoder for SegVol, ported to MLX.

Encodes points, boxes, masks, and text embeddings into sparse/dense prompts.
Uses sin/cos (2×) positional encoding (SegVol/VISTA3D style, NOT SAM-Med3D's sin/cos/sin 3×).
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 384, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = scale * mx.random.normal((3, num_pos_feats))

    def _pe_encoding(self, coords: mx.array) -> mx.array:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)

    def __call__(self, size: Tuple[int, int, int]) -> mx.array:
        """Dense PE for a grid. Returns (C, H, W, D)."""
        h, w, d = size
        grid = mx.ones((h, w, d))
        y_embed = mx.cumsum(grid, axis=0) - 0.5
        x_embed = mx.cumsum(grid, axis=1) - 0.5
        z_embed = mx.cumsum(grid, axis=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d
        pe = self._pe_encoding(mx.stack([x_embed, y_embed, z_embed], axis=-1))
        return pe.transpose(3, 0, 1, 2)

    def forward_with_coords(self, coords_input: mx.array,
                            image_size: Tuple[int, int, int]) -> mx.array:
        """Encode point coordinates. coords: (B, N, 3) in voxel space."""
        coords = mx.array(coords_input)
        # SegVol normalizes: coord[0]/image_size[1], coord[1]/image_size[0], coord[2]/image_size[2]
        coords_0 = coords[:, :, 0:1] / image_size[1]
        coords_1 = coords[:, :, 1:2] / image_size[0]
        coords_2 = coords[:, :, 2:3] / image_size[2]
        coords = mx.concatenate([coords_0, coords_1, coords_2], axis=-1)
        return self._pe_encoding(coords)


class PromptEncoder(nn.Module):
    """Encodes points, boxes, masks, and text into sparse/dense embeddings."""

    def __init__(self, embed_dim: int = 768,
                 image_embedding_size: Tuple[int, int, int] = (8, 16, 16),
                 input_image_size: Tuple[int, int, int] = (32, 256, 256),
                 mask_in_chans: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 4 point embeddings: positive, negative, box corner 1, box corner 2
        self.point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(4)]
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> mx.array:
        """Dense PE: (1, C, H, W, D) channels-first for mask decoder compatibility."""
        pe = self.pe_layer(self.image_embedding_size)  # (C, H, W, D)
        return pe[None]  # (1, C, H, W, D)

    def _embed_points(self, points: mx.array, labels: mx.array, pad: bool) -> mx.array:
        points = points + 0.5
        if pad:
            padding_point = mx.zeros((points.shape[0], 1, 3))
            padding_label = -mx.ones((labels.shape[0], 1))
            points = mx.concatenate([points, padding_point], axis=1)
            labels = mx.concatenate([labels, padding_label], axis=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        neg1_mask = (labels == -1)[:, :, None]
        zero_mask = (labels == 0)[:, :, None]
        one_mask = (labels == 1)[:, :, None]

        not_pt = self.not_a_point_embed(mx.zeros((1,), dtype=mx.int32))[0]
        neg_emb = self.point_embeddings[0](mx.zeros((1,), dtype=mx.int32))[0]
        pos_emb = self.point_embeddings[1](mx.zeros((1,), dtype=mx.int32))[0]

        point_embedding = mx.where(neg1_mask, not_pt, point_embedding)
        point_embedding = point_embedding + mx.where(zero_mask, neg_emb, mx.zeros_like(neg_emb))
        point_embedding = point_embedding + mx.where(one_mask, pos_emb, mx.zeros_like(pos_emb))
        return point_embedding

    def _embed_boxes(self, boxes: mx.array) -> mx.array:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 3)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding_0 = corner_embedding[:, 0:1, :] + self.point_embeddings[2](mx.zeros((1,), dtype=mx.int32))
        corner_embedding_1 = corner_embedding[:, 1:2, :] + self.point_embeddings[3](mx.zeros((1,), dtype=mx.int32))
        return mx.concatenate([corner_embedding_0, corner_embedding_1], axis=1)

    def __call__(
        self,
        points: Optional[Tuple[mx.array, mx.array]] = None,
        boxes: Optional[mx.array] = None,
        masks: Optional[mx.array] = None,
        text_embedding: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        bs = 1
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif text_embedding is not None:
            bs = text_embedding.shape[0]

        sparse_embeddings = mx.zeros((bs, 0, self.embed_dim))

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = mx.concatenate([sparse_embeddings, point_embeddings], axis=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = mx.concatenate([sparse_embeddings, box_embeddings], axis=1)

        if text_embedding is not None:
            # Text embedding as additional sparse token
            sparse_embeddings = mx.concatenate(
                [sparse_embeddings, text_embedding[:, None, :]], axis=1)

        if masks is not None:
            raise NotImplementedError("Mask prompts not yet implemented")
        else:
            no_mask = self.no_mask_embed(mx.zeros((1,), dtype=mx.int32))
            d, h, w = self.image_embedding_size
            dense_embeddings = mx.broadcast_to(
                no_mask.reshape(1, -1, 1, 1, 1),
                (bs, self.embed_dim, d, h, w))

        return sparse_embeddings, dense_embeddings
