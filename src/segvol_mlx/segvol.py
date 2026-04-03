"""Top-level SegVol model for MLX."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .image_encoder import ViTEncoder
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder
from .text_encoder import TextEncoder


class SegVol(nn.Module):
    """SegVol: text-prompted 3D medical image segmentation."""

    def __init__(
        self,
        image_encoder: ViTEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        text_encoder: Optional[TextEncoder] = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.text_encoder = text_encoder
        self._feat_shape = tuple(
            s // p for s, p in zip(image_encoder.img_size, image_encoder.patch_size)
        )

    def encode_image(self, x: mx.array) -> mx.array:
        """Encode image to spatial features.

        Args:
            x: (B, D, H, W, 1) channels-last CT volume
        Returns:
            (B, embed_dim, D', H', W') spatial features (channels-first for decoder)
        """
        tokens = self.image_encoder(x)  # (B, num_tokens, embed_dim)
        B = tokens.shape[0]
        C = tokens.shape[-1]
        # Reshape to spatial (channels-first for mask decoder)
        # (B, 2048, 768) → (B, 768, 8, 16, 16)
        return tokens.transpose(0, 2, 1).reshape(B, C, *self._feat_shape)

    def __call__(
        self,
        image: mx.array,
        text: Optional[mx.array] = None,
        text_embedding: Optional[mx.array] = None,
        boxes: Optional[mx.array] = None,
        points: Optional[Tuple[mx.array, mx.array]] = None,
        multimask_output: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            image: (B, D, H, W, 1) preprocessed CT volume, channels-last
            text: (B, seq_len) tokenized text prompt (requires text_encoder)
            text_embedding: (B, 768) pre-computed text embedding (bypasses text_encoder)
            boxes: (B, 6) bounding box [d1,h1,w1,d2,h2,w2]
            points: (coords: (B,N,3), labels: (B,N)) point prompts
            multimask_output: return 3 masks or 1

        Returns:
            masks: (B, num_masks, D', H', W') segmentation logits
            iou_pred: (B, num_masks) quality predictions
        """
        # Encode image
        image_embedding = self.encode_image(image)

        # Encode text if provided
        if text is not None and self.text_encoder is not None:
            text_embedding = self.text_encoder(text)

        # Encode prompts
        sparse, dense = self.prompt_encoder(
            points=points, boxes=boxes, text_embedding=text_embedding)

        dense_pe = self.prompt_encoder.get_dense_pe()

        # Decode masks
        masks, iou_pred = self.mask_decoder(
            image_embedding, dense_pe, sparse, dense,
            multimask_output=multimask_output,
            text_embedding=text_embedding)

        return masks, iou_pred

    def segment_by_text(
        self,
        image: mx.array,
        organ_name: str,
        tokenizer=None,
    ) -> mx.array:
        """Segment an organ using a text prompt.

        Args:
            image: (B, D, H, W, 1) preprocessed CT, channels-last
            organ_name: e.g. "liver", "spleen", "left lung upper lobe"
            tokenizer: CLIP tokenizer (load via text_encoder.load_tokenizer())

        Returns:
            mask: (B, 1, D', H', W') segmentation logits (threshold at 0)
        """
        from .text_encoder import SEGVOL_TEXT_TEMPLATE, tokenize_text

        if tokenizer is None:
            from .text_encoder import load_tokenizer
            tokenizer = load_tokenizer()

        text = SEGVOL_TEXT_TEMPLATE.format(organ_name)
        input_ids = tokenize_text(tokenizer, text)
        text_emb = self.text_encoder(input_ids)

        masks, _ = self(image, text_embedding=text_emb, multimask_output=False)
        return masks
