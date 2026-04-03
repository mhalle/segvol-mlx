"""SegVol inference pipeline with zoom-in-zoom-out for MLX."""

from typing import Optional, Tuple, List

import mlx.core as mx
import numpy as np
from scipy.ndimage import zoom


def preprocess_ct(
    volume: np.ndarray,
    target_size: Tuple[int, int, int] = (32, 256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a CT volume for SegVol.

    Args:
        volume: Raw CT array (D, H, W)
        target_size: Resize to this size

    Returns:
        resized: MinMax-normalized, resized to target_size
        scale: Scale factors used (for mapping coordinates back)
    """
    # MinMax normalize
    v_min, v_max = volume.min(), volume.max()
    vol_norm = (volume.astype(np.float32) - v_min) / (v_max - v_min + 1e-8)

    # Resize to target
    scale = np.array(target_size) / np.array(vol_norm.shape)
    resized = zoom(vol_norm, scale, order=1)

    return resized, scale


def logits2roi(spatial_size, logits, margin=5):
    """Extract ROI coordinates from coarse logits.

    Returns (min_d, min_h, min_w, max_d, max_h, max_w) or None if no foreground.
    """
    pred = (logits > 0).astype(np.uint8)
    if pred.sum() == 0:
        return None

    coords = np.argwhere(pred)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    # Add margin and clamp
    min_d = max(0, min_coords[0] - margin)
    min_h = max(0, min_coords[1] - margin)
    min_w = max(0, min_coords[2] - margin)
    max_d = min(logits.shape[0] - 1, max_coords[0] + margin)
    max_h = min(logits.shape[1] - 1, max_coords[1] + margin)
    max_w = min(logits.shape[2] - 1, max_coords[2] + margin)

    # Ensure minimum size
    for dim_min, dim_max, target_dim, shape_dim in [
        (min_d, max_d, spatial_size[0], logits.shape[0]),
        (min_h, max_h, spatial_size[1], logits.shape[1]),
        (min_w, max_w, spatial_size[2], logits.shape[2]),
    ]:
        if dim_max - dim_min + 1 < target_dim:
            center = (dim_min + dim_max) // 2
            half = target_dim // 2
            dim_min = max(0, center - half)
            dim_max = min(shape_dim - 1, dim_min + target_dim - 1)
            if dim_max - dim_min + 1 < target_dim:
                dim_min = max(0, dim_max - target_dim + 1)

    return min_d, min_h, min_w, max_d, max_h, max_w


def segment_at_point(
    model,
    volume: np.ndarray,
    organ_name: str,
    point_dhw: Tuple[int, int, int],
    spatial_size: Tuple[int, int, int] = (32, 256, 256),
    use_text: bool = True,
    use_zoom_in: bool = True,
) -> np.ndarray:
    """Segment an organ given a click point. Best single-call API.

    Extracts a slab centered on the click, runs point + optional text prompt,
    then optionally refines via zoom-in on the ROI.

    Args:
        model: Loaded SegVol model
        volume: Raw CT (D, H, W)
        organ_name: e.g. "liver", "spleen"
        point_dhw: Click location in voxel coordinates (d, h, w)
        spatial_size: Model input size
        use_text: Add text prompt (helps some organs, not others)
        use_zoom_in: Refine via ROI zoom-in

    Returns:
        Full-volume binary mask (D, H, W)
    """
    from .text_encoder import get_organ_embedding

    D, H, W = volume.shape
    v_min, v_max = volume.min(), volume.max()
    vol_norm = (volume.astype(np.float32) - v_min) / (v_max - v_min + 1e-8)

    slab_depth = spatial_size[0]
    d_start = max(0, point_dhw[0] - slab_depth // 2)
    d_end = min(D, d_start + slab_depth)
    if d_end - d_start < slab_depth:
        d_start = max(0, d_end - slab_depth)

    slab = vol_norm[d_start:d_end, :, :]
    scale = np.array(spatial_size) / np.array(slab.shape)
    slab_resized = zoom(slab, scale, order=1)

    # Map point to resized space
    local_point = (np.array(point_dhw) - np.array([d_start, 0, 0])) * scale
    coords = mx.array(local_point[None, None, :].astype(np.float32))
    labels = mx.array([[1.0]])

    # Optional text embedding
    text_emb = None
    if use_text:
        emb = get_organ_embedding(organ_name)
        if emb is not None:
            text_emb = emb[None]

    batch = mx.array(slab_resized[None, :, :, :, None])
    masks, _ = model(batch, points=(coords, labels), text_embedding=text_emb, multimask_output=False)
    mx.eval(masks)
    logits = np.array(masks)[0, 0]

    # Upscale to slab resolution
    slab_shape = (d_end - d_start, H, W)
    up_scale = np.array(slab_shape) / np.array(logits.shape)
    logits_full = zoom(logits, up_scale, order=1)

    if use_zoom_in and (logits_full > 0).sum() > 0:
        # Zoom-in: crop ROI and re-run
        roi = logits2roi(spatial_size, logits_full)
        if roi is not None:
            min_d, min_h, min_w, max_d, max_h, max_w = roi
            roi_crop = slab[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
            roi_scale = np.array(spatial_size) / np.array(roi_crop.shape)
            roi_resized = zoom(roi_crop, roi_scale, order=1)

            # Map point to ROI space
            roi_point = (local_point / scale - np.array([min_d, min_h, min_w])) * roi_scale
            roi_coords = mx.array(roi_point[None, None, :].astype(np.float32))

            batch_roi = mx.array(roi_resized[None, :, :, :, None])
            masks_roi, _ = model(batch_roi, points=(roi_coords, labels),
                                 text_embedding=text_emb, multimask_output=False)
            mx.eval(masks_roi)
            logits_roi = np.array(masks_roi)[0, 0]
            roi_up = np.array(roi_crop.shape) / np.array(logits_roi.shape)
            logits_roi_full = zoom(logits_roi, roi_up, order=1)
            logits_full[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1] = logits_roi_full

    # Embed in full volume
    result = np.zeros((D, H, W), dtype=np.uint8)
    result[d_start:d_end, :, :] = (logits_full > 0).astype(np.uint8)
    return result


def segment_slab(
    model,
    volume: np.ndarray,
    organ_name: str,
    center_slice: Optional[int] = None,
    slab_depth: int = 32,
    spatial_size: Tuple[int, int, int] = (32, 256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment an organ from a single slab of a CT volume.

    This is the simplest and fastest mode — extract a 32-slice slab,
    resize to (32, 256, 256), and run text-prompted segmentation.

    Args:
        model: Loaded SegVol model
        volume: Raw CT (D, H, W)
        organ_name: e.g. "liver", "spleen"
        center_slice: Center slice index. If None, uses middle of volume.
        slab_depth: Number of slices to extract
        spatial_size: Model input size

    Returns:
        mask: Binary mask at original slab resolution (slab_depth, H, W)
        slab_range: (d_start, d_end) indices into original volume
    """
    D, H, W = volume.shape
    v_min, v_max = volume.min(), volume.max()
    vol_norm = (volume.astype(np.float32) - v_min) / (v_max - v_min + 1e-8)

    if center_slice is None:
        center_slice = D // 2
    d_start = max(0, center_slice - slab_depth // 2)
    d_end = min(D, d_start + slab_depth)
    if d_end - d_start < slab_depth:
        d_start = max(0, d_end - slab_depth)

    slab = vol_norm[d_start:d_end, :, :]
    scale = np.array(spatial_size) / np.array(slab.shape)
    slab_resized = zoom(slab, scale, order=1)

    batch = mx.array(slab_resized[None, :, :, :, None])
    masks = model.segment_by_text(batch, organ_name)
    mx.eval(masks)
    logits = np.array(masks)[0, 0]

    # Upscale to original slab resolution
    up_scale = np.array(slab.shape) / np.array(logits.shape)
    mask = (zoom(logits, up_scale, order=1) > 0).astype(np.uint8)

    return mask, (d_start, d_end)


def sliding_window_segment(
    model,
    volume: np.ndarray,
    organ_name: str,
    point_dhw: Optional[Tuple[int, int, int]] = None,
    bbox: Optional[Tuple[int, int, int, int, int, int]] = None,
    spatial_size: Tuple[int, int, int] = (32, 256, 256),
    overlap: float = 0.5,
    verbose: bool = True,
) -> np.ndarray:
    """Full-volume segmentation via sliding window with point prompts.

    Uses overlapping 32-slice slabs with a point prompt projected into each slab.
    Point-prompted slabs produce much better results than text-only.

    For best results, provide either:
    - point_dhw: a click location (from user interaction or TotalSegmentator centroid)
    - bbox: a bounding box (d1,h1,w1,d2,h2,w2) to constrain the search

    Args:
        model: Loaded SegVol model
        volume: Raw CT (D, H, W) in HU
        organ_name: Used for text embedding (combined with point for best results)
        point_dhw: Click location in voxel coordinates
        bbox: Bounding box to constrain slabs (d1,h1,w1,d2,h2,w2)
        spatial_size: Model input size (32, 256, 256)
        overlap: Slab overlap fraction
        verbose: Print progress

    Returns:
        Binary mask (D, H, W)
    """
    from .text_encoder import get_organ_embedding
    from scipy.ndimage import gaussian_filter

    D, H, W = volume.shape
    v_min, v_max = volume.min(), volume.max()
    vol_norm = (volume.astype(np.float32) - v_min) / (v_max - v_min + 1e-8)

    slab_depth = spatial_size[0]
    step = max(1, int(slab_depth * (1 - overlap)))

    text_emb = None
    text_emb_np = get_organ_embedding(organ_name)
    if text_emb_np is not None:
        text_emb = text_emb_np[None]

    # Determine slab range
    if bbox is not None:
        d_min_range = max(0, bbox[0] - slab_depth)
        d_max_range = min(D, bbox[3] + slab_depth)
    elif point_dhw is not None:
        d_min_range = max(0, point_dhw[0] - slab_depth * 2)
        d_max_range = min(D, point_dhw[0] + slab_depth * 2)
    else:
        d_min_range, d_max_range = 0, D

    # Gaussian weighting along depth
    gauss_1d = np.zeros(slab_depth, dtype=np.float32)
    gauss_1d[slab_depth // 2] = 1.0
    gauss_1d = gaussian_filter(gauss_1d, sigma=slab_depth / 6)
    gauss_1d = gauss_1d / gauss_1d.max()
    gauss_weight = gauss_1d[:, None, None]

    logit_acc = np.zeros((D, H, W), dtype=np.float32)
    count_acc = np.zeros((D, H, W), dtype=np.float32)
    n_slabs = 0

    for d_start in range(d_min_range, max(d_min_range + 1, d_max_range - slab_depth + 1), step):
        d_end = min(d_start + slab_depth, D)
        if d_end - d_start < slab_depth:
            d_start = max(0, d_end - slab_depth)

        slab = vol_norm[d_start:d_end, :, :]
        scale = np.array(spatial_size) / np.array(slab.shape)
        slab_resized = zoom(slab, scale, order=1)
        batch = mx.array(slab_resized[None, :, :, :, None])

        # Build prompts for this slab
        kwargs = {"multimask_output": False}
        if text_emb is not None:
            kwargs["text_embedding"] = text_emb

        if point_dhw is not None and d_start <= point_dhw[0] < d_end:
            local_pt = (np.array(point_dhw) - np.array([d_start, 0, 0])) * scale
            kwargs["points"] = (
                mx.array(local_pt[None, None, :].astype(np.float32)),
                mx.array([[1.0]]),
            )

        if bbox is not None:
            local_box = np.array([
                max(0, (bbox[0] - d_start)) * scale[0],
                bbox[1] * scale[1],
                bbox[2] * scale[2],
                min(slab_depth - 1, (bbox[3] - d_start)) * scale[0],
                bbox[4] * scale[1],
                bbox[5] * scale[2],
            ], dtype=np.float32)
            kwargs["boxes"] = mx.array(local_box[None])

        masks, _ = model(batch, **kwargs)
        mx.eval(masks)
        logits = np.array(masks)[0, 0]

        actual_depth = d_end - d_start
        logits_full = zoom(logits, np.array((actual_depth, H, W)) / np.array(logits.shape), order=1)

        gw = gauss_weight[:actual_depth]
        logit_acc[d_start:d_end] += logits_full * gw
        count_acc[d_start:d_end] += gw
        n_slabs += 1

    logit_acc /= np.maximum(count_acc, 1e-8)

    if verbose:
        pos = (logit_acc > 0).sum()
        print(f"  {n_slabs} slabs, {pos} positive voxels")

    return (logit_acc > 0).astype(np.uint8)


def segment_organ(
    model,
    volume: np.ndarray,
    organ_name: str,
    spatial_size: Tuple[int, int, int] = (32, 256, 256),
    use_zoom_in: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Segment an organ from a CT volume using slab-based zoom-in-zoom-out.

    Strategy:
    1. Slide through volume in 32-slice slabs (50% overlap)
    2. For each slab: resize to (32, 256, 256), run text-prompted segmentation
    3. Accumulate best logits per voxel (argmax across slabs)
    4. Optional zoom-in: crop ROI from coarse result, re-run at full resolution

    Args:
        model: Loaded SegVol model
        volume: Raw CT volume (D, H, W) in HU
        organ_name: e.g. "liver", "spleen"
        spatial_size: Model input size
        use_zoom_in: Enable zoom-in refinement
        verbose: Print progress

    Returns:
        Binary segmentation mask (D, H, W) at original resolution
    """
    ori_shape = volume.shape
    D, H, W = ori_shape

    # MinMax normalize
    v_min, v_max = volume.min(), volume.max()
    vol_norm = (volume.astype(np.float32) - v_min) / (v_max - v_min + 1e-8)

    # --- Zoom-out: slab-based coverage ---
    slab_depth = spatial_size[0]  # 32
    step = slab_depth // 2  # 50% overlap
    best_logits = np.full(ori_shape, -np.inf, dtype=np.float32)

    n_slabs = 0
    for d_start in range(0, max(1, D - slab_depth + 1), step):
        d_end = min(d_start + slab_depth, D)
        if d_end - d_start < slab_depth:
            d_start = max(0, d_end - slab_depth)

        slab = vol_norm[d_start:d_end, :, :]
        scale = np.array(spatial_size) / np.array(slab.shape)
        slab_resized = zoom(slab, scale, order=1)

        batch = mx.array(slab_resized[None, :, :, :, None])
        masks = model.segment_by_text(batch, organ_name)
        mx.eval(masks)
        logits = np.array(masks)[0, 0]  # (32, 64, 64)

        # Upscale to slab's original resolution
        slab_shape = (d_end - d_start, H, W)
        up_scale = np.array(slab_shape) / np.array(logits.shape)
        logits_full = zoom(logits, up_scale, order=1)

        # Argmax accumulation
        region = best_logits[d_start:d_end, :, :]
        better = logits_full > region
        region[better] = logits_full[better]
        n_slabs += 1

    if verbose:
        zo_pred = (best_logits > 0).sum()
        print(f"  Zoom-out: {n_slabs} slabs, {zo_pred} positive voxels")

    if not use_zoom_in or (best_logits > 0).sum() == 0:
        return (best_logits > 0).astype(np.uint8)

    # --- Zoom-in: refine the ROI ---
    roi = logits2roi(spatial_size, best_logits)
    if roi is None:
        return (best_logits > 0).astype(np.uint8)

    min_d, min_h, min_w, max_d, max_h, max_w = roi
    roi_crop = vol_norm[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]

    if verbose:
        print(f"  Zoom-in: ROI {roi_crop.shape}")

    # Resize ROI to spatial_size and run again
    roi_scale = np.array(spatial_size) / np.array(roi_crop.shape)
    roi_resized = zoom(roi_crop, roi_scale, order=1)

    batch_roi = mx.array(roi_resized[None, :, :, :, None])
    masks_roi = model.segment_by_text(batch_roi, organ_name)
    mx.eval(masks_roi)
    logits_roi = np.array(masks_roi)[0, 0]

    roi_up_scale = np.array(roi_crop.shape) / np.array(logits_roi.shape)
    logits_roi_full = zoom(logits_roi, roi_up_scale, order=1)

    # Replace ROI region in global logits
    best_logits[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1] = logits_roi_full

    if verbose:
        zi_pred = (best_logits > 0).sum()
        print(f"  Zoom-in: {zi_pred} positive voxels")

    return (best_logits > 0).astype(np.uint8)
