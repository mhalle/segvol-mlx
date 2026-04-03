"""CLIP text encoder for SegVol, ported to MLX.

Minimal CLIP text model (12-layer transformer, 512 dim) with dim_align projection.
Supports precomputed embeddings for known organ classes.
"""

from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class CLIPTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 77


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.mlp_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.mlp_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = self.self_attn(self.layer_norm1(x), mask=mask)
        x = x + r
        r = self.mlp_fc2(nn.gelu_approx(self.mlp_fc1(self.layer_norm2(x))))
        x = x + r
        return x


class CLIPTextModel(nn.Module):
    """Minimal CLIP text encoder matching openai/clip-vit-base-patch32."""

    def __init__(self, config: Optional[CLIPTextConfig] = None):
        super().__init__()
        if config is None:
            config = CLIPTextConfig()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.encoder_layers = [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """
        Args:
            input_ids: (B, seq_len) token IDs
        Returns:
            pooler_output: (B, 512) — embedding at the EOT token position
        """
        B, N = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding.weight[:N]

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)

        for layer in self.encoder_layers:
            x = layer(x, mask=mask)

        x = self.final_layer_norm(x)

        # Pool at EOT token (highest token ID in each sequence)
        eot_tokens = mx.argmax(input_ids, axis=-1)
        pooler_output = x[mx.arange(B), eot_tokens]
        return pooler_output


class TextEncoder(nn.Module):
    """SegVol text encoder: CLIP text model + dimension alignment."""

    def __init__(self):
        super().__init__()
        self.clip_text_model = CLIPTextModel()
        self.dim_align = nn.Linear(512, 768)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """
        Args:
            input_ids: (B, seq_len) tokenized text
        Returns:
            text_embedding: (B, 768) aligned embedding
        """
        clip_output = self.clip_text_model(input_ids)  # (B, 512)
        return self.dim_align(clip_output)  # (B, 768)


# --- Precomputed embeddings for known organs ---

def load_tokenizer():
    """Load the CLIP tokenizer from HuggingFace SegVol repo."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = hf_hub_download("BAAI/SegVol", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_padding(length=77, pad_id=49407)
    tokenizer.enable_truncation(max_length=77)
    return tokenizer


def tokenize_text(tokenizer, text: str) -> mx.array:
    """Tokenize a text prompt for CLIP. Returns (1, 77) token IDs."""
    encoded = tokenizer.encode(text)
    return mx.array([encoded.ids])


SEGVOL_TEXT_TEMPLATE = "A computerized tomography of a {}."

# These organ names match SegVol's training data
SEGVOL_ORGAN_NAMES = [
    "liver", "right kidney", "spleen", "pancreas", "aorta",
    "inferior vena cava", "right adrenal gland", "left adrenal gland",
    "gallbladder", "esophagus", "stomach", "duodenum", "left kidney",
    "bladder", "prostate or uterus", "portal vein and splenic vein",
    "rectum", "small bowel", "lung",
    "left lung upper lobe", "left lung lower lobe",
    "right lung upper lobe", "right lung middle lobe", "right lung lower lobe",
    "trachea", "heart myocardium", "heart atrium left", "heart ventricle left",
    "heart atrium right", "heart ventricle right", "pulmonary artery",
    "brain", "iliac artery left", "iliac artery right",
    "iliac vena left", "iliac vena right",
    "hip left", "hip right", "sacrum",
    "vertebrae L5", "vertebrae L4", "vertebrae L3", "vertebrae L2", "vertebrae L1",
    "vertebrae T12", "vertebrae T11", "vertebrae T10", "vertebrae T9",
    "vertebrae T8", "vertebrae T7", "vertebrae T6", "vertebrae T5",
    "vertebrae T4", "vertebrae T3", "vertebrae T2", "vertebrae T1",
    "vertebrae C7", "vertebrae C6", "vertebrae C5", "vertebrae C4",
    "vertebrae C3", "vertebrae C2", "vertebrae C1",
    "colon", "kidney tumor", "liver tumor", "lung tumor", "pancreatic tumor",
]
