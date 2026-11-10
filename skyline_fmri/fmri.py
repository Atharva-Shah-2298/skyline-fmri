"""Functional MRI analysis used by Skyline's BOLD and activation-map layers.

Everything here is plain NumPy/SciPy so it can run in a background thread
while the viewer keeps rendering. Volumes are indexed ``(X, Y, Z)`` and time
series ``(X, Y, Z, T)``; world coordinates are RAS+ millimeters.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

from dipy.io.utils import split_filename_extension
from dipy.segment.threshold import otsu
from dipy.utils.logging import logger

NORMALIZATIONS = ("raw", "percent", "zscore")
STAT_SIGNS = ("both", "positive", "negative")

# Voxels scanned per block in voxelwise fits; bounds peak memory on big runs.
_CHUNK = 65536


def repetition_time(*, fname=None, img=None):
    """Return the repetition time, in seconds, of a BOLD run.

    The BIDS sidecar (``<name>.json`` next to the image) wins over the NIfTI
    header because scanners and converters often leave the header's time
    zoom unset.

    Parameters
    ----------
    fname : str or Path, optional
        Path of the NIfTI file; used to locate the BIDS sidecar.
    img : nibabel.spatialimages.SpatialImage, optional
        Loaded image whose fourth zoom is read when no sidecar is found.

    Returns
    -------
    float or None
        The repetition time in seconds, or None when neither source gives a
        positive value.
    """
    if fname is not None:
        stem, _ = split_filename_extension(str(fname))
        sidecar = Path(fname).with_name(f"{stem}.json")
        if sidecar.is_file():
            try:
                tr = float(json.loads(sidecar.read_text())["RepetitionTime"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Ignoring RepetitionTime in {sidecar}: {e}")
            else:
                if tr > 0:
                    return tr
    if img is None:
        return None
    zooms = img.header.get_zooms()
    if len(zooms) < 4 or not zooms[3] > 0:
        return None
    tr = float(zooms[3])
    time_unit = img.header.get_xyzt_units()[1]
    if time_unit == "msec":
        tr /= 1e3
    elif time_unit == "usec":
        tr /= 1e6
    return tr


def load_events(fname):
    """Read a BIDS events file into per-condition onset/duration arrays.

    Parameters
    ----------
    fname : str or Path
        Tab-separated (``.tsv``) or comma-separated (``.csv``) table with
        ``onset`` and ``duration`` columns in seconds. An optional
        ``trial_type`` column names the condition; rows without one belong to
        ``"task"``. Rows whose onset or duration is ``n/a`` are skipped.

    Returns
    -------
    dict
        Maps each condition name to an ``(N, 2)`` float array of
        ``(onset, duration)`` rows, sorted by onset, in first-seen order.

    Raises
    ------
    ValueError
        If the table lacks an ``onset`` or ``duration`` column.
    """
    fname = Path(fname)
    delimiter = "," if fname.suffix.lower() == ".csv" else "\t"
    rows = {}
    with fname.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        if "onset" not in columns or "duration" not in columns:
            raise ValueError(
                f"{fname.name} needs 'onset' and 'duration' columns; "
                f"found {list(columns)}."
            )
        trial_column = columns.get("trial_type")
        for row in reader:
            try:
                onset = float(row[columns["onset"]])
                duration = float(row[columns["duration"]])
            except (TypeError, ValueError):
                continue
            condition = (row.get(trial_column) or "").strip() if trial_column else ""
            if condition.lower() in ("", "n/a"):
                condition = "task"
            rows.setdefault(condition, []).append((onset, max(duration, 0.0)))
    return {
        condition: np.asarray(sorted(entries), dtype=float)
        for condition, entries in rows.items()
    }


def spm_hrf(dt, *, length=32.0):
    """Sample SPM's canonical double-gamma haemodynamic response.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds.
    length : float, optional
        Duration of the kernel in seconds.

    Returns
    -------
    ndarray
        Kernel samples at ``0, dt, 2 dt, ...`` normalized to unit sum, peaking
        near 5 s with an undershoot near 15 s.
    """
    t = np.arange(0.0, length, dt)
    hrf = stats.gamma.pdf(t, 6.0) - stats.gamma.pdf(t, 16.0) / 6.0
    return hrf / hrf.sum()


def task_regressors(events, n_scans, tr, *, oversampling=16):
    """Convolve event boxcars with the canonical HRF, sampled per volume.

    Parameters
    ----------
    events : dict
        Condition name to ``(N, 2)`` ``(onset, duration)`` array, as returned
        by :func:`load_events`. Zero-duration events are impulses.
    n_scans : int
        Number of volumes in the run.
    tr : float
        Repetition time in seconds.
    oversampling : int, optional
        Sub-TR samples used to build boxcars before convolution.

    Returns
    -------
    names : list of str
        Condition names in column order.
    regressors : ndarray, shape (n_scans, len(names))
        Predicted BOLD response for each condition, peak-normalized.
    """
    dt = tr / oversampling
    n_high = n_scans * oversampling
    hrf = spm_hrf(dt)
    names = list(events)
    regressors = np.zeros((n_scans, len(names)))
    for column, condition in enumerate(names):
        boxcar = np.zeros(n_high)
        for onset, duration in np.asarray(events[condition], dtype=float):
            start = int(round(onset / dt))
            stop = max(int(round((onset + duration) / dt)), start + 1)
            boxcar[max(start, 0) : max(min(stop, n_high), 0)] = 1.0
        response = np.convolve(boxcar, hrf)[:n_high][::oversampling]
        peak = np.abs(response).max()
        regressors[:, column] = response / peak if peak > 0 else response
    return names, regressors


def cosine_drift(n_scans, tr, *, cutoff=128.0):
    """Build an intercept plus a DCT high-pass basis.

    Parameters
    ----------
    n_scans : int
        Number of volumes.
    tr : float
        Repetition time in seconds.
    cutoff : float, optional
        Period in seconds below which slow drifts are modelled (SPM default).

    Returns
    -------
    ndarray, shape (n_scans, K + 1)
        Constant column followed by ``K`` cosines of increasing frequency.
    """
    order = min(int(np.floor(2.0 * n_scans * tr / cutoff)), n_scans - 1)
    t = np.arange(n_scans)[:, None]
    k = np.arange(1, order + 1)[None, :]
    cosines = np.sqrt(2.0 / n_scans) * np.cos(np.pi * (t + 0.5) * k / n_scans)
    return np.column_stack([np.ones(n_scans), cosines])


def design_matrix(events, n_scans, tr, *, cutoff=128.0):
    """Assemble task regressors and drift terms into one design.

    Parameters
    ----------
    events : dict
        Condition name to ``(N, 2)`` ``(onset, duration)`` array.
    n_scans : int
        Number of volumes.
    tr : float
        Repetition time in seconds.
    cutoff : float, optional
        High-pass period in seconds for the drift basis.

    Returns
    -------
    names : list of str
        Task condition names; they label the first ``len(names)`` columns.
    design : ndarray, shape (n_scans, P)
        Task regressors followed by the intercept and cosine drifts.
    """
    names, regressors = task_regressors(events, n_scans, tr)
    return names, np.column_stack(
        [regressors, cosine_drift(n_scans, tr, cutoff=cutoff)]
    )


def fit_glm(data, events, tr, *, mask=None, cutoff=128.0):
    """Fit a voxelwise OLS GLM and return one t-map per task condition.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, T)
        BOLD time series.
    events : dict
        Condition name to ``(N, 2)`` ``(onset, duration)`` array.
    tr : float
        Repetition time in seconds.
    mask : ndarray of bool, shape (X, Y, Z), optional
        Voxels to fit; all voxels when omitted. Unfitted voxels are zero.
    cutoff : float, optional
        High-pass period in seconds for the drift basis.

    Returns
    -------
    dict
        Condition name to a float32 ``(X, Y, Z)`` t-map of that condition's
        effect against the implicit baseline.

    Raises
    ------
    ValueError
        If ``events`` is empty or the design has no residual degrees of
        freedom.
    """
    if not events:
        raise ValueError("At least one task condition is required to fit a GLM.")
    n_scans = data.shape[-1]
    names, design = design_matrix(events, n_scans, tr, cutoff=cutoff)
    dof = n_scans - np.linalg.matrix_rank(design)
    if dof <= 0:
        raise ValueError("The design has no residual degrees of freedom.")
    pinv = np.linalg.pinv(design)
    unscaled_var = np.diag(pinv @ pinv.T)[: len(names)]
    mask = _spatial_mask(data, mask)
    voxels = np.flatnonzero(mask)
    flat = data.reshape(-1, n_scans)
    tvals = np.zeros((len(names), voxels.size), dtype=np.float32)
    for start in range(0, voxels.size, _CHUNK):
        block = voxels[start : start + _CHUNK]
        y = flat[block].astype(np.float64).T
        betas = pinv @ y
        residuals = y - design @ betas
        sigma2 = np.einsum("ij,ij->j", residuals, residuals) / dof
        scale = np.sqrt(np.outer(unscaled_var, sigma2))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_block = np.where(scale > 0, betas[: len(names)] / scale, 0.0)
        tvals[:, start : start + block.size] = t_block
    maps = {}
    for index, name in enumerate(names):
        volume = np.zeros(mask.size, dtype=np.float32)
        volume[voxels] = tvals[index]
        maps[name] = volume.reshape(mask.shape)
    return maps


def seed_correlation(data, seed, *, mask=None):
    """Correlate every voxel's time series with a seed time series.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, T)
        BOLD time series.
    seed : ndarray, shape (T,)
        Seed time series.
    mask : ndarray of bool, shape (X, Y, Z), optional
        Voxels to correlate; all voxels when omitted. Others are zero.

    Returns
    -------
    ndarray, shape (X, Y, Z)
        Pearson correlation as float32. Constant voxels, and every voxel when
        the seed is constant, are zero.
    """
    seed = np.asarray(seed, dtype=np.float64)
    seed = seed - seed.mean()
    seed_norm = np.linalg.norm(seed)
    mask = _spatial_mask(data, mask)
    result = np.zeros(mask.size, dtype=np.float32)
    if seed_norm == 0:
        return result.reshape(mask.shape)
    seed /= seed_norm
    voxels = np.flatnonzero(mask)
    flat = data.reshape(-1, data.shape[-1])
    for start in range(0, voxels.size, _CHUNK):
        block = voxels[start : start + _CHUNK]
        y = flat[block].astype(np.float64)
        y -= y.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(y, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(norms > 0, (y @ seed) / norms, 0.0)
        result[block] = r
    return result.reshape(mask.shape)


def epi_mask(data):
    """Estimate a brain mask from the temporal mean of a BOLD run.

    Pre-masked runs (many exact zeros) keep their non-zero support. Otherwise
    the mean is split with Otsu's threshold. Holes are filled and only the
    largest connected component is kept.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, T) or (X, Y, Z)
        BOLD time series, or its temporal mean.

    Returns
    -------
    ndarray of bool, shape (X, Y, Z)
        Brain mask.
    """
    mean = data.mean(axis=-1) if data.ndim == 4 else np.asarray(data)
    if np.mean(mean == 0) > 0.2:
        mask = mean != 0
    else:
        mask = mean > otsu(mean)
    labels, n_labels = ndimage.label(mask)
    if n_labels > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == np.argmax(sizes)
    return ndimage.binary_fill_holes(mask)


def percent_change(data, tr, *, mask=None, cutoff=128.0, smooth=0.0):
    """Return each volume's percent signal change after removing slow drift.

    Every voxel's series is regressed on an intercept plus a DCT high-pass
    basis; the residual is divided by the voxel's mean. This is the signal
    that task responses and resting fluctuations show up in, typically a few
    percent, whereas raw intensities are dominated by anatomy and drift.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, T)
        BOLD time series.
    tr : float
        Repetition time in seconds.
    mask : ndarray of bool, shape (X, Y, Z), optional
        Voxels to compute; all voxels when omitted. Others are zero.
    cutoff : float, optional
        High-pass period in seconds.
    smooth : float, optional
        Gaussian sigma, in voxels, applied to each volume within the mask
        (normalized so the brain edge is not darkened). 0 disables it.

    Returns
    -------
    ndarray, shape (X, Y, Z, T)
        Float32 percent change; zero outside the mask and where the mean is
        zero.
    """
    n_scans = data.shape[-1]
    drift = cosine_drift(n_scans, tr, cutoff=cutoff)
    projector = drift @ np.linalg.pinv(drift)
    mask = _spatial_mask(data, mask)
    voxels = np.flatnonzero(mask)
    flat = data.reshape(-1, n_scans)
    result = np.zeros(flat.shape, dtype=np.float32)
    for start in range(0, voxels.size, _CHUNK):
        block = voxels[start : start + _CHUNK]
        y = flat[block].astype(np.float64)
        mean = y.mean(axis=1, keepdims=True)
        residual = y - y @ projector.T
        with np.errstate(divide="ignore", invalid="ignore"):
            result[block] = np.where(mean != 0, 100.0 * residual / mean, 0.0)
    result = result.reshape(data.shape)
    if smooth > 0:
        weight = ndimage.gaussian_filter(mask.astype(np.float32), smooth)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(mask, 1.0 / weight, 0.0).astype(np.float32)
        for frame in range(n_scans):
            result[..., frame] = (
                ndimage.gaussian_filter(result[..., frame], smooth) * norm
            )
    return result


def temporal_snr(data):
    """Return the voxelwise temporal mean divided by temporal standard deviation.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, T)
        BOLD time series.

    Returns
    -------
    ndarray, shape (X, Y, Z)
        Float32 tSNR; zero where the standard deviation is zero.
    """
    mean = data.mean(axis=-1, dtype=np.float64)
    std = data.std(axis=-1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std, 0.0).astype(np.float32)


def voxel_from_world(world, affine, shape):
    """Return the voxel holding a world point, or None outside the volume.

    Parameters
    ----------
    world : array-like, shape (3,)
        RAS+ coordinate in millimeters.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    shape : tuple of int
        Spatial shape ``(X, Y, Z)``.

    Returns
    -------
    tuple of int or None
        Nearest voxel index, or None when it falls outside ``shape``.
    """
    ijk = np.linalg.inv(affine) @ np.r_[np.asarray(world, dtype=float)[:3], 1.0]
    ijk = np.rint(ijk[:3]).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= np.asarray(shape[:3])):
        return None
    return tuple(int(v) for v in ijk)


def normalize_timeseries(ts, *, mode="raw", detrend=False):
    """Rescale a time series for display.

    Parameters
    ----------
    ts : array-like, shape (T,)
        Time series.
    mode : {"raw", "percent", "zscore"}, optional
        ``"percent"`` gives percent change from the mean; ``"zscore"``
        subtracts the mean and divides by the standard deviation.
    detrend : bool, optional
        Remove a least-squares linear trend first. The mean used for percent
        change is taken before detrending, so the scale stays meaningful.

    Returns
    -------
    ndarray, shape (T,)
        The rescaled series as float64.

    Raises
    ------
    ValueError
        If ``mode`` is not one of :data:`NORMALIZATIONS`.
    """
    if mode not in NORMALIZATIONS:
        raise ValueError(f"mode must be one of {NORMALIZATIONS}, got {mode!r}.")
    ts = np.asarray(ts, dtype=np.float64)
    mean = ts.mean()
    if detrend and ts.size > 1:
        t = np.arange(ts.size)
        slope, intercept = np.polyfit(t, ts, 1)
        ts = ts - (slope * t + intercept) + mean
    if mode == "percent":
        return 100.0 * (ts - mean) / mean if mean != 0 else ts - mean
    if mode == "zscore":
        std = ts.std()
        return (ts - ts.mean()) / std if std > 0 else ts - ts.mean()
    return ts


def power_spectrum(ts, tr):
    """Return the one-sided power spectrum of a demeaned time series.

    Parameters
    ----------
    ts : array-like, shape (T,)
        Time series.
    tr : float
        Repetition time in seconds.

    Returns
    -------
    freqs : ndarray
        Frequencies in Hz, excluding DC.
    power : ndarray
        Periodogram power at ``freqs``.
    """
    ts = np.asarray(ts, dtype=np.float64)
    spectrum = np.fft.rfft(ts - ts.mean())
    freqs = np.fft.rfftfreq(ts.size, d=tr)
    return freqs[1:], (np.abs(spectrum) ** 2 / ts.size)[1:]


def fitted_response(ts, events, tr, *, cutoff=128.0):
    """Project a time series onto the task design and return the model fit.

    Parameters
    ----------
    ts : array-like, shape (T,)
        Voxel time series.
    events : dict
        Condition name to ``(N, 2)`` ``(onset, duration)`` array.
    tr : float
        Repetition time in seconds.
    cutoff : float, optional
        High-pass period in seconds for the drift basis.

    Returns
    -------
    ndarray, shape (T,)
        Least-squares fit of the full design (task plus drifts).
    """
    ts = np.asarray(ts, dtype=np.float64)
    _, design = design_matrix(events, ts.size, tr, cutoff=cutoff)
    betas, *_ = np.linalg.lstsq(design, ts, rcond=None)
    return design @ betas


def threshold_clusters(stat, threshold, *, min_size=0, sign="both", affine=None):
    """Remove sub-threshold voxels and small clusters from a statistical map.

    Clusters are 26-connected groups of voxels beyond ``threshold``; positive
    and negative clusters are labelled separately.

    Parameters
    ----------
    stat : ndarray, shape (X, Y, Z)
        Statistical map (t, z, r, ...).
    threshold : float
        Magnitude a voxel must reach to belong to a cluster.
    min_size : int, optional
        Clusters with fewer voxels are removed.
    sign : {"both", "positive", "negative"}, optional
        Which tail(s) to keep.
    affine : ndarray, shape (4, 4), optional
        Voxel-to-world affine used to report peak coordinates in mm.

    Returns
    -------
    thresholded : ndarray, shape (X, Y, Z)
        ``stat`` with every voxel outside a kept cluster set to zero.
    clusters : list of dict
        One entry per kept cluster, strongest peak first, with ``label`` (its
        value in ``labels``), ``size`` (voxels), ``peak`` (signed value),
        ``peak_index`` (voxel) and, when ``affine`` is given, ``peak_world``
        (mm).
    labels : ndarray of int32, shape (X, Y, Z)
        Kept cluster labels, unique across both tails; 0 elsewhere.

    Raises
    ------
    ValueError
        If ``sign`` is not one of :data:`STAT_SIGNS`.
    """
    if sign not in STAT_SIGNS:
        raise ValueError(f"sign must be one of {STAT_SIGNS}, got {sign!r}.")
    stat = np.asarray(stat, dtype=np.float32)
    keep = np.zeros(stat.shape, dtype=bool)
    cluster_labels = np.zeros(stat.shape, dtype=np.int32)
    next_label = 1
    clusters = []
    structure = ndimage.generate_binary_structure(3, 3)
    tails = []
    if sign in ("both", "positive"):
        tails.append(stat >= threshold)
    if sign in ("both", "negative"):
        tails.append(stat <= -threshold)
    for supra in tails:
        labels, n_labels = ndimage.label(supra, structure=structure)
        if n_labels == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
        kept = sizes >= max(min_size, 1)
        kept[0] = False
        keep |= kept[labels]
        index = np.flatnonzero(kept)
        if index.size == 0:
            continue
        relabel = np.zeros(n_labels + 1, dtype=np.int32)
        relabel[index] = np.arange(next_label, next_label + index.size)
        cluster_labels += relabel[labels]
        extreme = ndimage.maximum_position(np.abs(stat), labels, index)
        for label, position in zip(index, extreme):
            entry = {
                "label": int(relabel[label]),
                "size": int(sizes[label]),
                "peak": float(stat[position]),
                "peak_index": tuple(int(v) for v in position),
            }
            if affine is not None:
                entry["peak_world"] = (np.asarray(affine) @ np.r_[position, 1.0])[:3]
            clusters.append(entry)
        next_label += index.size
    clusters.sort(key=lambda entry: -abs(entry["peak"]))
    return np.where(keep, stat, 0).astype(np.float32), clusters, cluster_labels


def _spatial_mask(data, mask):
    """Return a boolean spatial mask matching ``data``'s first three axes."""
    if mask is None:
        return np.ones(data.shape[:3], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data shape {data.shape[:3]}."
        )
    return mask
