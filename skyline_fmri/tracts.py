"""Link white-matter tracts to activation clusters and to the cursor.

Tracts are identified by the stem of their file name, as in DIPY's HCP842
bundle atlas (``AF_L.trk`` is the left arcuate fasciculus). Streamlines are
taken in world (RAS+ mm) coordinates, the space Skyline renders them in.
"""

from pathlib import Path
import re

import numpy as np

#: Full names for the abbreviations used by the HCP842 bundle atlases.
TRACT_NAMES = {
    "AC": "Anterior commissure",
    "AF": "Arcuate fasciculus",
    "AR": "Acoustic radiation",
    "AST": "Frontal aslant tract",
    "C": "Cingulum",
    "CB": "Cerebellum",
    "CC": "Corpus callosum",
    "CCMid": "Corpus callosum (midbody)",
    "CC_ForcepsMajor": "Forceps major",
    "CC_ForcepsMinor": "Forceps minor",
    "CNII": "Optic nerve (CN II)",
    "CNIII": "Oculomotor nerve (CN III)",
    "CNIV": "Trochlear nerve (CN IV)",
    "CNV": "Trigeminal nerve (CN V)",
    "CNVII": "Facial nerve (CN VII)",
    "CNVIII": "Vestibulocochlear nerve (CN VIII)",
    "CS": "Corticostriatal tract",
    "CST": "Corticospinal tract",
    "CT": "Corticothalamic tract",
    "CTT": "Central tegmental tract",
    "DLF": "Dorsal longitudinal fasciculus",
    "EMC": "Extreme capsule",
    "F_L_R": "Fornix",
    "FPT": "Frontopontine tract",  # codespell:ignore
    "ICP": "Inferior cerebellar peduncle",
    "IFOF": "Inferior fronto-occipital fasciculus",
    "ILF": "Inferior longitudinal fasciculus",
    "LL": "Lateral lemniscus",
    "MCP": "Middle cerebellar peduncle",
    "MdLF": "Middle longitudinal fasciculus",
    "ML": "Medial lemniscus",
    "MLF": "Medial longitudinal fasciculus",
    "OPT": "Occipitopontine tract",
    "OR": "Optic radiation",
    "PC": "Posterior commissure",
    "PPT": "Parietopontine tract",
    "RST": "Rubrospinal tract",
    "SCP": "Superior cerebellar peduncle",
    "SLF": "Superior longitudinal fasciculus",
    "STT": "Spinothalamic tract",
    "TPT": "Temporopontine tract",
    "UF": "Uncinate fasciculus",
    "V": "Vermis",
    "VOF": "Vertical occipital fasciculus",
}

_SIDES = {"L": "left", "R": "right"}


def tract_key(path):
    """Return the atlas key of a tract file (its name without extensions).

    Parameters
    ----------
    path : str or Path
        Tractogram path or display name, such as ``"bundles/AF_L.trk"``.

    Returns
    -------
    str
        The file stem, such as ``"AF_L"``.
    """
    return Path(str(path)).name.split(".")[0]


def tract_label(path):
    """Return a readable tract name for an atlas file name.

    Parameters
    ----------
    path : str or Path
        Tractogram path or display name.

    Returns
    -------
    str
        ``"Arcuate fasciculus (left)"`` for ``AF_L.trk``; unknown names are
        returned as their file stem.
    """
    key = tract_key(path)
    if key in TRACT_NAMES:
        return TRACT_NAMES[key]
    match = re.fullmatch(r"(.+)_([LR])", key)
    if match and match.group(1) in TRACT_NAMES:
        return f"{TRACT_NAMES[match.group(1)]} ({_SIDES[match.group(2)]})"
    return key


def _points_and_owner(streamlines):
    """Stack streamline points and record which streamline each came from."""
    lengths = np.array([len(s) for s in streamlines], dtype=np.int64)
    if lengths.size == 0 or lengths.sum() == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), lengths
    points = np.concatenate([np.asarray(s, dtype=np.float64) for s in streamlines])
    owner = np.repeat(np.arange(lengths.size), lengths)
    return points, owner, lengths


def _labels_at(points, labels, affine):
    """Return the label of the voxel holding each point (0 outside)."""
    ijk = np.rint(
        (np.linalg.inv(affine) @ np.c_[points, np.ones(len(points))].T)[:3].T
    ).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(labels.shape)), axis=1)
    values = np.zeros(len(points), dtype=labels.dtype)
    values[inside] = labels[tuple(ijk[inside].T)]
    return values


def streamlines_through(streamlines, labels, affine, *, label=None):
    """Flag the streamlines that pass through labelled voxels.

    Parameters
    ----------
    streamlines : sequence of ndarray
        Streamlines in world coordinates.
    labels : ndarray of int, shape (X, Y, Z)
        Cluster labels, 0 for background.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine of ``labels``.
    label : int, optional
        Only count this label; any non-zero label when omitted.

    Returns
    -------
    ndarray of bool
        One flag per streamline.
    """
    points, owner, lengths = _points_and_owner(streamlines)
    if points.size == 0:
        return np.zeros(len(lengths), dtype=bool)
    values = _labels_at(points, labels, affine)
    hit = values == label if label is not None else values != 0
    return np.bincount(owner[hit], minlength=len(lengths)) > 0


def tract_cluster_overlap(labels, affine, tracts):
    """Count, for each cluster, the streamlines of each tract passing through.

    Parameters
    ----------
    labels : ndarray of int, shape (X, Y, Z)
        Cluster labels, 0 for background.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine of ``labels``.
    tracts : dict
        Tract name to a sequence of world-space streamlines.

    Returns
    -------
    per_cluster : dict
        Cluster label to a list of ``(name, count, fraction)`` tuples, most
        streamlines first; ``fraction`` is relative to the whole tract.
    overall : list of tuple
        ``(name, count, fraction)`` for streamlines crossing any cluster.
    """
    per_cluster = {}
    overall = []
    for name, streamlines in tracts.items():
        points, owner, lengths = _points_and_owner(streamlines)
        if points.size == 0:
            continue
        values = _labels_at(points, labels, affine)
        hit = values != 0
        if not hit.any():
            continue
        total = len(lengths)
        crossing = np.unique(owner[hit])
        overall.append((name, int(crossing.size), crossing.size / total))
        pairs = np.unique(np.c_[values[hit], owner[hit]], axis=0)
        cluster_ids, counts = np.unique(pairs[:, 0], return_counts=True)
        for cluster_id, count in zip(cluster_ids, counts):
            per_cluster.setdefault(int(cluster_id), []).append(
                (name, int(count), count / total)
            )
    for entries in per_cluster.values():
        entries.sort(key=lambda entry: -entry[1])
    overall.sort(key=lambda entry: -entry[1])
    return per_cluster, overall


class TractIndex:
    """Answer which tracts pass near a world point.

    Streamline points are binned on a regular grid once; lookups test the
    bins within ``radius`` of the query point.

    Parameters
    ----------
    tracts : dict
        Tract name to a sequence of world-space streamlines.
    spacing : float, optional
        Grid spacing in millimeters.
    """

    def __init__(self, tracts, *, spacing=1.0):
        """Bin every tract's points.

        Parameters
        ----------
        tracts : dict
            Tract name to a sequence of world-space streamlines.
        spacing : float, optional
            Grid spacing in millimeters.
        """
        self.spacing = float(spacing)
        self._bins = {}
        for name, streamlines in tracts.items():
            points, _, _ = _points_and_owner(streamlines)
            if points.size:
                self._bins[name] = np.unique(self._codes(points))

    def _codes(self, points):
        """Hash grid cells of ``points`` into int64 codes."""
        cells = np.floor(np.asarray(points) / self.spacing).astype(np.int64)
        cells += 1 << 20
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    def tracts_at(self, world, *, radius=2.0):
        """Return the tracts with points within ``radius`` mm of ``world``.

        Parameters
        ----------
        world : array-like, shape (3,)
            Query point in world millimeters.
        radius : float, optional
            Search radius in millimeters.

        Returns
        -------
        list of str
            Tract names, in the order they were indexed.
        """
        steps = int(np.ceil(radius / self.spacing))
        offsets = np.stack(
            np.meshgrid(*[np.arange(-steps, steps + 1)] * 3, indexing="ij"), -1
        ).reshape(-1, 3)
        offsets = offsets[
            np.linalg.norm(offsets, axis=1) * self.spacing <= radius + 1e-9
        ]
        query = self._codes(
            np.asarray(world, dtype=float)[None, :3] + offsets * self.spacing
        )
        return [
            name for name, codes in self._bins.items() if np.isin(query, codes).any()
        ]
