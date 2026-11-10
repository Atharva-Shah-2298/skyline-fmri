"""Masks for glass-brain views: a translucent brain shell and 3D clusters."""

import numpy as np
from scipy import ndimage

from dipy.segment.threshold import otsu

POSITIVE_COLOR = (0.98, 0.42, 0.1)
NEGATIVE_COLOR = (0.15, 0.5, 1.0)


def brain_mask(volume, *, closing=2):
    """Estimate a solid mask of the brain, or the head, from an anatomy.

    The volume is split from its background with Otsu's threshold; the
    largest connected object is kept, closed and filled. Skull-stripped images
    give a brain mask, full-head images give a head mask.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z) or (X, Y, Z, N)
        Anatomical image; the first volume is used for 4D data.
    closing : int, optional
        Iterations of morphological closing that bridge sulci and small gaps.

    Returns
    -------
    ndarray of bool, shape (X, Y, Z)
        The mask.

    Raises
    ------
    ValueError
        If the volume has no positive voxel.
    """
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.where(np.isfinite(data), data, 0)
    if not (data > 0).any():
        raise ValueError("The volume has no positive voxel to build a brain mask.")
    mask = data > otsu(data)
    labels, n_objects = ndimage.label(mask)
    if n_objects > 1:
        sizes = ndimage.sum(mask, labels, range(1, n_objects + 1))
        mask = labels == np.argmax(sizes) + 1
    if closing:
        mask = ndimage.binary_closing(mask, iterations=closing, border_value=0)
    return ndimage.binary_fill_holes(mask)


def cluster_masks(display):
    """Split a thresholded statistical map into its positive and negative parts.

    Parameters
    ----------
    display : ndarray, shape (X, Y, Z)
        Statistical map with sub-threshold voxels set to zero.

    Returns
    -------
    list of tuple
        ``(sign, mask, color)`` for each non-empty tail, positive first, with
        ``sign`` either ``"positive"`` or ``"negative"``.
    """
    display = np.nan_to_num(np.asarray(display, dtype=np.float32))
    tails = (
        ("positive", display > 0, POSITIVE_COLOR),
        ("negative", display < 0, NEGATIVE_COLOR),
    )
    return [(sign, mask, color) for sign, mask, color in tails if mask.any()]
