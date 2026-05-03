"""SegVol MLX: Text-prompted 3D medical image segmentation on Apple Silicon."""

from .segvol import SegVol
from .weights import download_and_load, load_segvol
from .inference import (
    preprocess_ct,
    normalize_ct,
    foreground_normalize,
    minmax_normalize,
)
from .text_encoder import (
    get_organ_embedding,
    list_organs,
    load_tokenizer,
    SEGVOL_TEXT_TEMPLATE,
    SEGVOL_ORGAN_NAMES,
)

__all__ = [
    "SegVol",
    "download_and_load",
    "load_segvol",
    "preprocess_ct",
    "normalize_ct",
    "foreground_normalize",
    "minmax_normalize",
    "get_organ_embedding",
    "list_organs",
    "load_tokenizer",
    "SEGVOL_TEXT_TEMPLATE",
    "SEGVOL_ORGAN_NAMES",
]
