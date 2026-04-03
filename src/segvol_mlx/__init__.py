"""SegVol MLX: Text-prompted 3D medical image segmentation on Apple Silicon."""

from .weights import download_and_load, load_segvol
from .text_encoder import (
    get_organ_embedding,
    list_organs,
    load_tokenizer,
    SEGVOL_TEXT_TEMPLATE,
    SEGVOL_ORGAN_NAMES,
)
