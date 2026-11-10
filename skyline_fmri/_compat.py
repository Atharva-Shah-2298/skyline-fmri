"""Helpers from DIPY's development branch that DIPY 1.12 does not ship.

They are copied from DIPY (BSD 3-Clause license) so Skyline fMRI runs with the
released ``dipy`` package.
"""

from functools import partial

import nibabel as nib
import numpy as np
from scipy import sparse

from dipy.utils.optpkg import optional_package

fury, has_fury, _ = optional_package("fury", min_version="2.1.0")
if has_fury:
    from fury.actor import Group, create_axes_helper, surface
    from fury.colormap import (
        create_colormap as fury_create_colormap,
        hsv2rgb,
        normalize_colors,
    )
    from fury.lib import EventType
    from fury.utils import apply_affine, voxel_mesh_by_object
    from fury.window import update_camera
else:
    Group = create_axes_helper = surface = fury_create_colormap = hsv2rgb = fury
    normalize_colors = EventType = apply_affine = voxel_mesh_by_object = fury
    update_camera = fury

_RGB_FIELD_NAMES = frozenset({("R", "G", "B"), ("R", "G", "B", "A")})


def has_rgb_dtype(data):
    """Check whether an array has a structured NIfTI ``DT_RGB24`` dtype.

    Returns ``True`` when *data* has a structured numpy dtype whose field
    names are exactly ``('R', 'G', 'B')`` or ``('R', 'G', 'B', 'A')``,
    as created by nibabel when loading a NIfTI file with datatype code
    128 (``DT_RGB24``) or 2304 (``DT_RGBA32``).

    Parameters
    ----------
    data : np.ndarray
        Array to inspect.

    Returns
    -------
    bool
        ``True`` when *data* carries a structured RGB/RGBA dtype.
    """
    return data.dtype.names is not None and tuple(data.dtype.names) in _RGB_FIELD_NAMES


def is_rgb_compatible_data(data):
    """Check whether a plain numeric array has RGB-compatible shape and values.

    Returns ``True`` when *data* has ``ndim == 4``, the last dimension is
    3 or 4, the array is non-empty, and values fall within a range
    consistent with color channels:

    - **uint8**: always compatible (no DWI scanner uses uint8).
    - **Other integer** (e.g. int16): all values in [0, 255].
    - **Float**: all values in [0, 1] with no NaN.

    Structured-dtype arrays (e.g. NIfTI ``DT_RGB24``) are not checked
    here; use `has_rgb_dtype` for those.

    Parameters
    ----------
    data : np.ndarray
        Array to inspect.

    Returns
    -------
    bool
        ``True`` when *data* has RGB-compatible shape and values.
    """
    if data.ndim != 4 or data.shape[-1] not in (3, 4):
        return False
    if data.size == 0:
        return False
    if data.dtype == np.uint8:
        return True
    if data.dtype.kind in ("u", "i"):
        return int(data.min()) >= 0 and int(data.max()) <= 255
    if data.dtype.kind == "f":
        if np.isnan(data).any():
            return False
        return float(data.min()) >= 0.0 and float(data.max()) <= 1.0
    return False


def unpack_rgb_array(data):
    """Convert a structured RGB/RGBA array to a plain numeric array.

    Nibabel represents RGB/RGBA NIfTI images with datatype code 128
    (``DT_RGB24``) as structured arrays, e.g.
    ``dtype=[('R','u1'),('G','u1'),('B','u1')]`` with shape
    ``(X, Y, Z)``.  NumPy ufuncs do not support structured dtypes, so this
    converts to a plain ``(X, Y, Z, N)`` array where *N* is the number of
    channels.

    Only arrays whose field names are exactly ``('R', 'G', 'B')`` or
    ``('R', 'G', 'B', 'A')`` are unpacked.  Other structured dtypes are
    returned unchanged.

    Parameters
    ----------
    data : np.ndarray
        Array that may have a structured RGB/RGBA dtype.

    Returns
    -------
    np.ndarray
        Regular numeric array when input was structured RGB/RGBA;
        otherwise the input unchanged.
    """
    if has_rgb_dtype(data):
        return np.stack([data[name] for name in data.dtype.names], axis=-1)
    return data


def get_orientation_from_affine(affine):
    """Get the orientation of the affine transformation matrix.

    Parameters
    ----------
    affine : ndarray
        The affine transformation matrix of shape (4, 4).

    Returns
    -------
    orientation : str
        The orientation of the affine transformation matrix.
    """
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a (4, 4) numpy array")
    return "".join(nib.aff2axcodes(affine))


def create_colormap(v, *, name="plasma", auto=True, hsv_range=None):
    """Create RGB colors from a named colormap or from HSV ramps.

    Parameters
    ----------
    v : array_like
        Values mapped to colors.
    name : str, optional
        Name of the colormap, as accepted by ``fury.colormap.create_colormap``.
        Ignored when ``hsv_range`` is given.
    auto : bool, optional
        If True, ``v`` is rescaled to [0, 1] from ``v.min()`` to ``v.max()``.
        Otherwise ``v`` is clipped to [0, 1].
    hsv_range : tuple of three (low, high) pairs, optional
        Hue, saturation and value ramps, each in [0, 1], from ``v = 0`` to
        ``v = 1``. Replaces ``name`` when given.

    Returns
    -------
    ndarray of shape (..., 3)
        RGB colors of ``v``.

    Raises
    ------
    ValueError
        If ``hsv_range`` does not have three (low, high) pairs.
    """
    if hsv_range is None:
        return fury_create_colormap(v, name=name, auto=auto)
    if len(hsv_range) != 3:
        raise ValueError(
            f"hsv_range must hold hue, saturation and value ramps, got {hsv_range}."
        )
    v = np.asarray(v, dtype=float)
    if auto:
        v = np.interp(v, [v.min(), v.max()], [0, 1])
    else:
        v = np.clip(v, 0, 1)
    hsv = np.stack([np.interp(v, [0, 1], ramp) for ramp in hsv_range], axis=-1)
    return hsv2rgb(hsv)


def scalars_to_lut_indices(values, clim, n_colors):
    """Map scalar values to the nearest index of an evenly spaced lookup table.

    Parameters
    ----------
    values : array_like
        Scalar values. Values outside ``clim``, including infinities, are
        clipped to the first or last index.
    clim : tuple of float
        Finite scalar values ``(low, high)`` mapped to the first and last index.
    n_colors : int
        Number of entries of the lookup table.

    Returns
    -------
    ndarray of int
        Lookup-table indices, with the shape of ``values``.

    Raises
    ------
    ValueError
        If ``clim`` is not finite or ``low > high``, if ``n_colors`` is lower
        than 1, or if ``values`` contains NaN.

    Notes
    -----
    When ``low == high`` every value is mapped to the first index.
    """
    low, high = clim
    if not (np.isfinite(low) and np.isfinite(high) and low <= high):
        raise ValueError(f"clim must be finite with low <= high, got {clim}.")
    if n_colors < 1:
        raise ValueError(f"n_colors must be at least 1, got {n_colors}.")
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("values must not contain NaN.")
    if low == high:
        return np.zeros(values.shape, dtype=np.intp)
    fraction = np.clip((values - low) / (high - low), 0, 1)
    return np.rint(fraction * (n_colors - 1)).astype(np.intp)


def _sync_gizmo(parts, camera):
    """Orient the gizmo and fade its axes to match the camera.

    Parameters
    ----------
    parts : dict
        Output of ``fury.actor.create_axes_helper``.
    camera : fury.lib.PerspectiveCamera
        Screen camera the gizmo follows.
    """
    x, y, z, w = camera.world.rotation
    parts["group"].local.rotation = np.array([x, y, -z, w], dtype=np.float32)

    billboard = np.array([-x, -y, z, w], dtype=np.float32)
    parts["center_disk"].local.rotation = billboard
    for disk, label in zip(parts["disks"], parts["labels"]):
        disk.local.rotation = billboard
        label.local.rotation = billboard

    depths = [float(np.dot(camera.world.forward, d)) for d in parts["axis_vectors"]]
    front = min(depths)
    span = max(max(depths) - front, 1e-6)
    for depth, disk, label, line in zip(
        depths, parts["disks"], parts["labels"], parts["lines"]
    ):
        opacity = 1.0 - (depth - front) / span * 0.9
        disk.opacity = opacity
        label.opacity = opacity
        line.opacity = opacity


def _snap_camera_to_axis(screen, direction):
    """Point the screen camera at the scene from an anatomical side.

    Parameters
    ----------
    screen : fury.window.Screen
        Screen whose camera is moved.
    direction : array_like
        World direction from the scene center toward the camera.
    """
    direction = np.asarray(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    camera = screen.camera
    bbox = screen.scene.main_scene.get_world_bounding_box()
    if bbox is not None and np.all(np.isfinite(bbox)):
        target = bbox.mean(axis=0)
    else:
        target = camera.world.position + camera.world.forward
    distance = max(float(np.linalg.norm(camera.world.position - target)), 1.0)
    up = (0, 1, 0) if abs(direction[2]) > 0.9 else (0, 0, 1)
    camera.world.reference_up = up
    camera.world.position = target + direction * distance
    camera.look_at(target)
    update_camera(camera, None, screen.scene)


def _on_gizmo_axis_pointer_down(event, *, screen, parts, direction):
    """Snap the camera to the clicked gizmo axis."""
    event.stop_propagation()
    _snap_camera_to_axis(screen, direction)
    _sync_gizmo(parts, screen.camera)


def gizmo(
    show_manager,
    *,
    screen=0,
    labels=("L", "R", "P", "A", "S", "I"),
    size=30,
    thickness=2,
    position=(60, 60),
):
    """Add an orientation gizmo with anatomical camera snapping.

    Parameters
    ----------
    show_manager : fury.window.ShowManager
        Window the gizmo is attached to.
    screen : int
        Index of the screen of ``show_manager`` that hosts the gizmo.
    labels : sequence of str
        Axis labels for the world directions -X, +X, -Y, +Y, +Z, -Z. The
        defaults are anatomical labels for RAS space.
    size : float
        Gizmo size in pixels.
    thickness : float
        Axis line thickness.
    position : tuple of float
        Gizmo center in pixels from the bottom-left of the screen viewport.

    Returns
    -------
    anchor : fury.actor.Group
        The gizmo actor, hosted in the screen's UI scene. Its ``endpoints``
        attribute maps each label to its clickable disk.

    Notes
    -----
    Clicking an axis label snaps the camera to that side of the scene, with
    superior up for L/R/A/P views and anterior up for S/I views.
    """
    name = f"dipy_gizmo_screen_{screen}"
    if name in show_manager.callbacks:
        raise ValueError(f"A gizmo is already attached to screen {screen}.")
    target_screen = show_manager.screens[screen]

    parts = create_axes_helper(labels=list(labels), thickness=thickness)
    anchor = Group(name="DIPY Gizmo")
    anchor.local.position = [position[0], position[1], 0.5]
    anchor.local.scale = [size, size, 0.45]
    anchor.add(parts["group"])

    for disk, label, direction in zip(
        parts["disks"], parts["labels"], parts["axis_vectors"]
    ):
        handler = partial(
            _on_gizmo_axis_pointer_down,
            screen=target_screen,
            parts=parts,
            direction=direction,
        )
        disk.add_event_handler(handler, EventType.POINTER_DOWN)
        label.add_event_handler(handler, EventType.POINTER_DOWN)

    anchor.endpoints = dict(zip(labels, parts["disks"]))
    target_screen.scene.ui_scene.add(anchor)
    _sync_gizmo(parts, target_screen.camera)
    show_manager.register_callback(
        _sync_gizmo, 0.016, True, name, parts, target_screen.camera
    )
    return anchor


def _taubin_smooth(vertices, faces, iterations, *, lam=0.5, mu=-0.53):
    """Smooth a triangle mesh with Taubin (shrink-free) Laplacian smoothing.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Mesh vertices.
    faces : ndarray, shape (M, 3)
        Triangle vertex indices.
    iterations : int
        Number of smoothing passes.
    lam : float, optional
        Positive (shrinking) step.
    mu : float, optional
        Negative (inflating) step, slightly larger in magnitude than ``lam``.

    Returns
    -------
    ndarray, shape (N, 3)
        Smoothed vertices.
    """
    n_vertices = len(vertices)
    rows = faces[:, [0, 1, 2, 1, 2, 0]].ravel()
    cols = faces[:, [1, 2, 0, 0, 1, 2]].ravel()
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices)
    ).tocsr()
    adjacency.data[:] = 1
    neighbor_mean = sparse.diags(1 / np.ravel(adjacency.sum(axis=1))) @ adjacency

    smoothed = np.array(vertices, dtype=float)
    for _ in range(iterations):
        smoothed += lam * (neighbor_mean @ smoothed - smoothed)
        smoothed += mu * (neighbor_mean @ smoothed - smoothed)
    return smoothed


def contour_from_roi(
    data,
    *,
    affine=None,
    color=(1, 0, 0),
    opacity=0.5,
    material="phong",
    iterations=20,
):
    """Generate a smooth surface actor from a binary ROI.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        ROI volume; voxels with a value above zero are displayed.
    affine : ndarray, shape (4, 4), optional
        Affine mapping voxel coordinates to world coordinates.
    color : str, tuple, list or ndarray, optional
        Contour color (hex string, RGB in [0, 1] or in [0, 255]).
    opacity : float, optional
        Opacity from 0 (transparent) to 1 (opaque).
    material : str, optional
        Material of the contour mesh, as accepted by :func:`fury.actor.surface`
        (``"phong"``, ``"basic"``, ``"standard"``, ``"physical"``).
    iterations : int, optional
        Number of smoothing passes. Use 0 to keep the blocky voxel surface.

    Returns
    -------
    Group
        One surface actor per connected object of the ROI.

    Notes
    -----
    Overrides :func:`fury.actor.contour_from_roi`, which draws staircased voxel
    faces. The surface is smoothed with Taubin smoothing, which does not shrink
    the ROI, and shaded with per-vertex normals so it does not look faceted.
    """
    color = normalize_colors(color)[0][:3]
    inside = np.asarray(data) > 0
    contour = Group()
    for mesh in voxel_mesh_by_object(data, triangulate=True).values():
        faces = _outward_faces(mesh["verts"], mesh["faces"], inside)
        # FURY puts voxel i between i and i + 1; voxel centers are at i.
        vertices = _taubin_smooth(mesh["verts"] - 0.5, faces, iterations)
        if affine is not None:
            vertices = apply_affine(affine, vertices)
            if np.linalg.det(np.asarray(affine)[:3, :3]) < 0:
                faces = faces[:, ::-1]
        actor = surface(
            vertices,
            faces,
            colors=color,
            opacity=opacity,
            material=material,
            normals=vertex_normals(vertices, faces),
        )
        actor.material.flat_shading = False
        contour.add(actor)
    return contour


def _outward_faces(vertices, faces, inside):
    """Wind the triangles of a voxel surface so their normals point outward.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Voxel-corner vertices, voxel ``i`` spanning ``[i, i + 1]``.
    faces : ndarray, shape (M, 3)
        Triangle vertex indices.
    inside : ndarray of bool, shape (X, Y, Z)
        Voxels enclosed by the surface.

    Returns
    -------
    ndarray, shape (M, 3)
        Faces, reversed where their normal pointed into ``inside``.
    """
    faces = np.array(faces)
    corners = np.asarray(vertices, dtype=np.float64)[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    probe = np.floor(corners.mean(axis=1) + 0.5 * normals).astype(int)
    valid = np.all((probe >= 0) & (probe < inside.shape), axis=1)
    inward = np.zeros(len(faces), dtype=bool)
    inward[valid] = inside[tuple(probe[valid].T)]
    faces[inward] = faces[inward][:, ::-1]
    return faces


def vertex_normals(vertices, faces):
    """Compute area-weighted unit normals at the vertices of a triangle mesh.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Mesh vertices.
    faces : ndarray, shape (M, 3)
        Triangle vertex indices.

    Returns
    -------
    ndarray, shape (N, 3)
        Unit normals, following the winding of ``faces``; zero for vertices
        without a face.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    corners = vertices[faces]
    face_normals = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.where(length > 0, length, 1)).astype(np.float32)
