"""Thresholded statistical maps drawn as activation overlays."""

import time

import nibabel as nib
import numpy as np

from dipy.utils.optpkg import optional_package
from skyline_fmri.UI.elements import segmented_switch, thin_slider
from skyline_fmri.UI.theme import THEME
from skyline_fmri.fmri import STAT_SIGNS, threshold_clusters
from skyline_fmri.render.image import Image3D
from skyline_fmri.render.renderer import format_affine_info
from skyline_fmri.tracts import tract_cluster_overlap, tract_key, tract_label

fury_trip_msg = (
    "Skyline requires Fury version 2.1.0 or higher."
    " Please upgrade Fury by `pip install -U fury` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.1.0",
    trip_msg=fury_trip_msg,
)
if has_fury_v2:
    from fury.actor import set_group_opacity
    from fury.lib import gfx

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui
    icons_fontawesome_6 = imgui_bundle.icons_fontawesome_6

# Color ramps from the threshold (first anchor) to the maximum (last anchor).
_HOT = ((0.75, 0.05, 0.02), (1.0, 0.45, 0.0), (1.0, 0.85, 0.1), (1.0, 1.0, 0.75))
_COLD = ((0.0, 0.15, 0.85), (0.0, 0.5, 1.0), (0.2, 0.85, 1.0), (0.75, 1.0, 1.0))

#: Positive ramp, negative ramp (None mirrors the positive one) per colormap.
STAT_COLORMAPS = {
    "Cold-Hot": (_HOT, _COLD),
    "Red-Yellow": (_HOT, None),
    "Blue-Cyan": (_COLD, None),
    "Viridis": ("viridis", None),
    "Plasma": ("plasma", None),
    "Inferno": ("inferno", None),
}

#: Number of lookup-table entries spanning ``[-vmax, vmax]``.
LUT_SIZE = 1024

_SIGN_LABELS = {"Both": "both", "Pos": "positive", "Neg": "negative"}

# Seconds a threshold must stay still before clusters are relabelled.
_CLUSTER_DEBOUNCE = 0.25


def _ramp(spec, t):
    """Sample a color ramp at fractions ``t`` in ``[0, 1]``.

    Parameters
    ----------
    spec : tuple or str
        Anchor colors, or the name of a ``pygfx`` colormap.
    t : ndarray
        Positions along the ramp.

    Returns
    -------
    ndarray, shape (len(t), 3)
        RGB colors.
    """
    if isinstance(spec, str):
        anchors = np.asarray(getattr(gfx.cm, spec).texture.data, dtype=float)[:, :3]
    else:
        anchors = np.asarray(spec, dtype=float)
    xp = np.linspace(0.0, 1.0, anchors.shape[0])
    return np.stack([np.interp(t, xp, anchors[:, c]) for c in range(3)], axis=1)


def stat_lut(colormap, threshold, vmax, sign, *, size=LUT_SIZE):
    """Build the RGBA lookup table of a thresholded statistical overlay.

    Entries map evenly onto ``[-vmax, vmax]``. Values whose magnitude is
    below ``threshold``, or whose sign is not shown, are fully transparent.

    Parameters
    ----------
    colormap : str
        Key of :data:`STAT_COLORMAPS`.
    threshold : float
        Smallest displayed magnitude.
    vmax : float
        Magnitude at which the ramp saturates.
    sign : {"both", "positive", "negative"}
        Which tail(s) are shown.
    size : int, optional
        Number of entries.

    Returns
    -------
    ndarray, shape (size, 4)
        Float32 RGBA table.
    """
    positive, negative = STAT_COLORMAPS[colormap]
    negative = positive if negative is None else negative
    values = (np.arange(size) + 0.5) / size * 2.0 * vmax - vmax
    magnitude = np.abs(values)
    span = max(vmax - threshold, 1e-12)
    t = np.clip((magnitude - threshold) / span, 0.0, 1.0)
    lut = np.zeros((size, 4), dtype=np.float32)
    pos = values >= 0
    lut[pos, :3] = _ramp(positive, t[pos])
    lut[~pos, :3] = _ramp(negative, t[~pos])
    visible = magnitude >= threshold
    if sign == "positive":
        visible &= pos
    elif sign == "negative":
        visible &= ~pos
    lut[:, 3] = visible
    return lut


def default_threshold(vmax):
    """Pick a starting display threshold for a map whose robust max is ``vmax``.

    t/z maps (``vmax >= 5``) start at 3.1 (p < 0.001, one-sided); weaker maps
    such as correlations start at 40% of their maximum.

    Parameters
    ----------
    vmax : float
        Robust maximum magnitude of the map.

    Returns
    -------
    float
        Starting threshold.
    """
    return 3.1 if vmax >= 5.0 else round(0.4 * vmax, 3)


def create_stat_map_visualization(
    input,
    idx,
    *,
    render_callback=None,
    sync_callabck=None,
    navigate_callback=None,
    tract_provider=None,
    isolate_callback=None,
    threshold=None,
    colormap="Cold-Hot",
):
    """Create a StatMap3D overlay from an already-loaded statistical map.

    Parameters
    ----------
    input : tuple
        ``(data, affine)`` or ``(data, affine, filename)`` where ``data`` is a
        3D ndarray or nibabel image. The filename is a display label.
    idx : int
        Index used to name the map when no filename is given.
    render_callback : callable, optional
        Callback used to request a render/update.
    sync_callabck : callable, optional
        Callback used to synchronize slice positions across visualizations.
    navigate_callback : callable, optional
        Called as ``navigate_callback(viz, world)`` when a cluster peak is
        chosen in the cluster table.
    threshold : float, optional
        Initial display threshold; see :func:`default_threshold` when None.
    colormap : str, optional
        Key of :data:`STAT_COLORMAPS`.
    tract_provider : callable, optional
        Returns the loaded tracts as ``{name: streamlines}`` (world mm).
    isolate_callback : callable, optional
        Called as ``isolate_callback(viz, name)`` to isolate a tract's
        streamlines crossing the activation.

    Returns
    -------
    StatMap3D
        The created overlay.

    Raises
    ------
    ValueError
        If the input is not a tuple of length 2 or 3.
    """
    if not isinstance(input, tuple) or len(input) not in (2, 3):
        raise ValueError(
            "Input must be a tuple containing (data, affine, filename) or "
            "(data, affine) for statistical map visualization."
        )
    data, affine, *rest = input
    filename = rest[0] if rest else f"StatMap_{idx}"
    return StatMap3D(
        filename,
        data,
        affine=affine,
        render_callback=render_callback,
        sync_callabck=sync_callabck,
        navigate_callback=navigate_callback,
        tract_provider=tract_provider,
        isolate_callback=isolate_callback,
        threshold=threshold,
        colormap=colormap,
    )


class StatMap3D(Image3D):
    """A thresholded statistical map (t, z, r, ...) overlaid on anatomy.

    Voxels below the threshold are transparent; the rest are colored by
    magnitude with separate ramps for positive and negative values. The
    overlay blends on top of opaque slices at the same position, so it reads
    as an activation map over whatever anatomy or diffusion image is loaded.
    Clusters smaller than ``min_cluster_size`` voxels are hidden and the
    remaining clusters are listed by peak.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    volume : nibabel.spatialimages.SpatialImage or ndarray
        3D map (singleton fourth axes are squeezed).
    affine : ndarray, optional
        Voxel-to-world affine; uses the image affine when omitted.
    threshold : float, optional
        Smallest displayed magnitude; see :func:`default_threshold`.
    vmax : float, optional
        Magnitude at which colors saturate; the 99.9th percentile of the
        non-zero magnitudes when omitted.
    sign : {"both", "positive", "negative"}, optional
        Which tail(s) are shown.
    min_cluster_size : int, optional
        Clusters with fewer voxels are hidden.
    colormap : str, optional
        Key of :data:`STAT_COLORMAPS`.
    opacity : int, optional
        Overlay opacity in percent.
    interpolation : str, optional
        Slice interpolation (``"linear"`` or ``"nearest"``).
    render_callback : callable, optional
        Callback used to request a render/update.
    sync_callabck : callable, optional
        Callback used to synchronize state across views.
    navigate_callback : callable, optional
        Called as ``navigate_callback(viz, world)`` when a cluster peak is
        chosen in the cluster table.
    """

    def __init__(
        self,
        name,
        volume,
        *,
        affine=None,
        threshold=None,
        vmax=None,
        sign="both",
        min_cluster_size=0,
        colormap="Cold-Hot",
        opacity=100,
        interpolation="linear",
        render_callback=None,
        sync_callabck=None,
        navigate_callback=None,
        tract_provider=None,
        isolate_callback=None,
    ):
        """Initialize the overlay.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        volume : nibabel.spatialimages.SpatialImage or ndarray
            3D map (singleton fourth axes are squeezed).
        affine : ndarray, optional
            Voxel-to-world affine; uses the image affine when omitted.
        threshold : float, optional
            Smallest displayed magnitude; see :func:`default_threshold`.
        vmax : float, optional
            Magnitude at which colors saturate.
        sign : {"both", "positive", "negative"}, optional
            Which tail(s) are shown.
        min_cluster_size : int, optional
            Clusters with fewer voxels are hidden.
        colormap : str, optional
            Key of :data:`STAT_COLORMAPS`.
        opacity : int, optional
            Overlay opacity in percent.
        interpolation : str, optional
            Slice interpolation (``"linear"`` or ``"nearest"``).
        render_callback : callable, optional
            Callback used to request a render/update.
        sync_callabck : callable, optional
            Callback used to synchronize state across views.
        navigate_callback : callable, optional
            Called with ``(viz, world)`` when a cluster peak is chosen.
        tract_provider : callable, optional
            Returns the loaded tracts as ``{name: streamlines}`` (world mm);
            used to list the tracts passing through each cluster.
        isolate_callback : callable, optional
            Called as ``isolate_callback(viz, name)`` to show only the
            streamlines of tract ``name`` that cross the activation.

        Raises
        ------
        ValueError
            If the map is not 3D, ``sign`` is unknown or ``colormap`` is not
            a key of :data:`STAT_COLORMAPS`.
        """
        if isinstance(volume, nib.spatialimages.SpatialImage):
            affine = volume.affine if affine is None else affine
            volume = volume.get_fdata(dtype=np.float32)
        stat = np.squeeze(np.nan_to_num(np.asarray(volume, dtype=np.float32)))
        if stat.ndim != 3:
            raise ValueError(f"Statistical maps must be 3D; got shape {stat.shape}.")
        if sign not in STAT_SIGNS:
            raise ValueError(f"sign must be one of {STAT_SIGNS}, got {sign!r}.")
        if colormap not in STAT_COLORMAPS:
            raise ValueError(
                f"colormap must be one of {tuple(STAT_COLORMAPS)}, got {colormap!r}."
            )
        self.stat = stat
        self.affine = affine
        self._max_abs = float(np.abs(stat).max()) or 1.0
        self.vmax = float(vmax) if vmax else self._robust_max(stat)
        self.threshold = (
            float(threshold) if threshold is not None else default_threshold(self.vmax)
        )
        self.sign = sign
        self.min_cluster_size = int(min_cluster_size)
        self._navigate_callback = navigate_callback
        self._tract_provider = tract_provider
        self._isolate_callback = isolate_callback
        self._clusters_due = None
        self.tract_overlap = {}
        self.tracts_through = []
        display, self.clusters, self.labels = self._threshold()
        super().__init__(
            name,
            display,
            affine=affine,
            interpolation=interpolation,
            render_callback=render_callback,
            opacity=opacity,
            rgb=False,
            colormap=colormap,
            sync_callabck=sync_callabck,
        )
        self._colormap_options = tuple(STAT_COLORMAPS)
        self._set_opacity(opacity)

    @staticmethod
    def _robust_max(stat):
        """Return the 99.9th percentile of non-zero magnitudes (1.0 if none)."""
        magnitudes = np.abs(stat[stat != 0])
        if magnitudes.size == 0:
            return 1.0
        return float(np.percentile(magnitudes, 99.9)) or 1.0

    @property
    def viz_type(self):
        """Sidebar group identifier for statistical overlays.

        Returns
        -------
        str
            ``"stat_map"``.
        """
        return "stat_map"

    @property
    def is_overlay(self):
        """Statistical maps are always transparent overlays.

        Returns
        -------
        bool
            True; the viewer stacks overlays above opaque images.
        """
        return True

    def _threshold(self):
        """Return the cluster-filtered map and its cluster table."""
        return threshold_clusters(
            self.stat,
            self.threshold,
            min_size=self.min_cluster_size,
            sign=self.sign,
            affine=self.affine,
        )

    def _populate_info(self):
        """Build the info tooltip with shape, range and affine.

        Returns
        -------
        str
            Multi-line description of the map.
        """
        info = (
            f"Dimensions: {self.stat.shape}\n"
            f"Range: [{self.stat.min():.3g}, {self.stat.max():.3g}]"
        )
        if self.affine is not None:
            info += "\n" + format_affine_info(self.affine)
        return info

    def _apply_colormap(self, colormap):
        """Apply the thresholded lookup table and overlay blending.

        Parameters
        ----------
        colormap : str
            Key of :data:`STAT_COLORMAPS`.
        """
        self.colormap = colormap
        self.value_range = (-self.vmax, self.vmax)
        lut = stat_lut(colormap, self.threshold, self.vmax, self.sign)
        texture_map = gfx.cm.create_colormap(lut, n=lut.shape[0])
        for actor in self._slicer.children:
            actor.material.map = texture_map
            actor.material.clim = self.value_range
            actor.material.alpha_mode = "blend"
            actor.material.depth_write = False
        # Drawn after opaque slices; the viewer lifts it toward the camera.
        self._slicer.render_order = 10

    def _set_opacity(self, opacity):
        """Set overlay opacity while keeping blended, non-depth-writing slices.

        Parameters
        ----------
        opacity : int
            Opacity in percent, in ``[0, 100]``.
        """
        set_group_opacity(self._slicer, opacity / 100.0)

    def update_threshold(
        self, *, threshold=None, vmax=None, sign=None, min_cluster_size=None
    ):
        """Change display thresholds and relabel clusters immediately.

        Parameters
        ----------
        threshold : float, optional
            New smallest displayed magnitude.
        vmax : float, optional
            New saturation magnitude.
        sign : {"both", "positive", "negative"}, optional
            New tail selection.
        min_cluster_size : int, optional
            New smallest kept cluster, in voxels.
        """
        if threshold is not None:
            self.threshold = float(threshold)
        if vmax is not None:
            self.vmax = float(vmax)
        if sign is not None:
            if sign not in STAT_SIGNS:
                raise ValueError(f"sign must be one of {STAT_SIGNS}, got {sign!r}.")
            self.sign = sign
        if min_cluster_size is not None:
            self.min_cluster_size = int(min_cluster_size)
        self.apply_scene_op(self._apply_colormap, self.colormap)
        self._relabel_clusters()

    def set_map(self, stat):
        """Replace the map with one of the same shape, keeping display settings.

        Parameters
        ----------
        stat : ndarray, shape (X, Y, Z)
            New statistical map.

        Raises
        ------
        ValueError
            If ``stat`` does not match the current shape.
        """
        stat = np.nan_to_num(np.asarray(stat, dtype=np.float32))
        if stat.shape != self.stat.shape:
            raise ValueError(
                f"New map shape {stat.shape} does not match {self.stat.shape}."
            )
        self.stat = stat
        self._max_abs = float(np.abs(stat).max()) or 1.0
        self._info = self._populate_info()
        self._relabel_clusters()

    def _relabel_clusters(self):
        """Recompute clusters, their tracts, and upload the filtered map."""
        self._clusters_due = None
        self.dwi, self.clusters, self.labels = self._threshold()
        self.refresh_tracts()
        self.apply_scene_op(self._upload_active_volume)

    def refresh_tracts(self):
        """Recount the streamlines of every loaded tract crossing each cluster."""
        tracts = self._tract_provider() if self._tract_provider is not None else {}
        if not tracts:
            self.tract_overlap, self.tracts_through = {}, []
            return
        affine = np.eye(4) if self.affine is None else self.affine
        self.tract_overlap, self.tracts_through = tract_cluster_overlap(
            self.labels, affine, tracts
        )

    def _schedule_relabel(self):
        """Relabel clusters once the controls have been still for a moment."""
        self._clusters_due = time.perf_counter() + _CLUSTER_DEBOUNCE

    def tick(self, now):
        """Run a pending debounced cluster relabel.

        Parameters
        ----------
        now : float
            Current ``time.perf_counter()`` value.
        """
        if self._clusters_due is not None and now >= self._clusters_due:
            self._relabel_clusters()

    def render_widgets(self):
        """Draw opacity, slices, threshold, colormap and cluster controls."""
        self._render_sync_controls()
        self._render_opacity_control()
        imgui.spacing()
        self._render_slice_controls()
        imgui.spacing()
        self._render_threshold_controls()
        imgui.spacing()
        self._render_colormap_control()
        imgui.spacing()
        self.draw_colorbar(imgui.get_window_draw_list())
        imgui.spacing()
        self._render_cluster_table()
        if self.tracts_through:
            imgui.spacing()
            self._render_tract_table()
        imgui.spacing()
        self._render_voxel_info()
        imgui.spacing()
        imgui.spacing()
        self._render_interpolation_control()
        imgui.spacing()

    def _render_threshold_controls(self):
        """Draw threshold, saturation, sign and cluster-size controls."""
        changed, value = thin_slider(
            "Threshold",
            self.threshold,
            0.0,
            self._max_abs,
            text_format=".2f",
            step=self._max_abs / 200.0,
        )
        if changed:
            self.threshold = min(float(value), self.vmax)
            self.apply_scene_op(self._apply_colormap, self.colormap)
            self._schedule_relabel()

        changed, value = thin_slider(
            "Max",
            self.vmax,
            self.threshold,
            max(self._max_abs, self.threshold + 1e-6),
            text_format=".2f",
            step=self._max_abs / 200.0,
        )
        if changed:
            self.vmax = max(float(value), self.threshold + 1e-6)
            self.apply_scene_op(self._apply_colormap, self.colormap)

        sign_label = next(k for k, v in _SIGN_LABELS.items() if v == self.sign)
        changed, label = segmented_switch("Show", list(_SIGN_LABELS), sign_label)
        if changed:
            self.sign = _SIGN_LABELS[label]
            self.apply_scene_op(self._apply_colormap, self.colormap)
            self._schedule_relabel()

        changed, value = thin_slider(
            "Min cluster",
            self.min_cluster_size,
            0,
            500,
            value_type="int",
            text_format=".0f",
            value_unit=" vox",
            step=1,
        )
        if changed:
            self.min_cluster_size = int(value)
            self._schedule_relabel()

    def _render_cluster_table(self):
        """List the strongest clusters; selecting a row jumps to its peak."""
        icon = icons_fontawesome_6.ICON_FA_LIST_OL
        imgui.text_colored(THEME["text"], f"{icon}  Clusters ({len(self.clusters)})")
        if not self.clusters:
            return
        flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
        )
        imgui.push_style_color(imgui.Col_.table_row_bg_alt, THEME["shadow"])
        imgui.push_style_color(imgui.Col_.header_hovered, THEME["shadow"])
        if imgui.begin_table("clusters", 5, flags):
            for header in ("Voxels", "Peak", "x, y, z (mm)", "Tracts", ""):
                imgui.table_setup_column(header)
            imgui.push_style_color(imgui.Col_.text, THEME["text"])
            imgui.table_headers_row()
            imgui.pop_style_color()
            for row, cluster in enumerate(self.clusters[:8]):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                clicked, _ = imgui.selectable(
                    f"{cluster['size']}##cluster{row}",
                    False,
                    imgui.SelectableFlags_.span_all_columns,
                )
                imgui.table_set_column_index(1)
                color = THEME["primary"] if cluster["peak"] > 0 else (0.3, 0.7, 1, 1)
                imgui.text_colored(color, f"{cluster['peak']:.2f}")
                imgui.table_set_column_index(2)
                world = cluster.get("peak_world", cluster["peak_index"])
                imgui.text_colored(
                    THEME["secondary"], ", ".join(f"{v:.0f}" for v in world)
                )
                imgui.table_set_column_index(3)
                tracts = self.tract_overlap.get(cluster["label"], [])
                names = ", ".join(tract_key(name) for name, _, _ in tracts[:2])
                if len(tracts) > 2:
                    names += f" +{len(tracts) - 2}"
                imgui.text_colored(THEME["secondary"], names or "-")
                if tracts and imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "\n".join(
                            f"{tract_label(name)}: {count} streamlines"
                            for name, count, _ in tracts
                        )
                    )
                imgui.table_set_column_index(4)
                imgui.text_colored(
                    THEME["text"], icons_fontawesome_6.ICON_FA_LOCATION_CROSSHAIRS
                )
                if clicked and self._navigate_callback is not None:
                    self._navigate_callback(self, np.asarray(world, dtype=float))
            imgui.end_table()
        imgui.pop_style_color(2)

    def _render_tract_table(self):
        """List tracts crossing the activation; the filter icon isolates one."""
        icon = icons_fontawesome_6.ICON_FA_ROUTE
        imgui.text_colored(
            THEME["text"],
            f"{icon}  Tracts through activation ({len(self.tracts_through)})",
        )
        flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
        )
        imgui.push_style_color(imgui.Col_.table_row_bg_alt, THEME["shadow"])
        if imgui.begin_table("tracts_through", 4, flags):
            for header in ("Tract", "Streamlines", "Of tract", ""):
                imgui.table_setup_column(header)
            imgui.push_style_color(imgui.Col_.text, THEME["text"])
            imgui.table_headers_row()
            imgui.pop_style_color()
            for row, (name, count, fraction) in enumerate(self.tracts_through[:12]):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text_colored(THEME["secondary"], tract_label(name))
                imgui.table_set_column_index(1)
                imgui.text_colored(THEME["primary"], str(count))
                imgui.table_set_column_index(2)
                imgui.text_colored(THEME["secondary"], f"{100 * fraction:.0f}%")
                imgui.table_set_column_index(3)
                imgui.push_id(f"isolate{row}")
                imgui.text_colored(THEME["text"], icons_fontawesome_6.ICON_FA_FILTER)
                if imgui.is_item_hovered():
                    imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
                    imgui.set_tooltip(
                        "Add a layer with only the streamlines crossing the activation"
                    )
                if imgui.is_item_clicked() and self._isolate_callback is not None:
                    self._isolate_callback(self, name)
                imgui.pop_id()
            imgui.end_table()
        imgui.pop_style_color()

    def draw_colorbar(self, draw_list, *, origin=None, size=None, vertical=False):
        """Draw this map's color scale with threshold and saturation labels.

        Parameters
        ----------
        draw_list : ImDrawList
            Target draw list.
        origin : tuple of float, optional
            Top-left corner in screen pixels; the current cursor when omitted,
            in which case layout space is reserved for the bar.
        size : tuple of float, optional
            Bar ``(width, height)``; full content width by 10 px when omitted.
        vertical : bool, optional
            Draw from ``-vmax`` (bottom) to ``vmax`` (top) instead of left to
            right.
        """
        advance = origin is None
        if origin is None:
            cursor = imgui.get_cursor_screen_pos()
            origin = (cursor.x, cursor.y)
        if size is None:
            size = (imgui.get_content_region_avail().x - 10, 10.0)
        x0, y0 = origin
        width, height = size
        lut = stat_lut(self.colormap, self.threshold, self.vmax, self.sign, size=128)
        hidden = imgui.get_color_u32(THEME["shadow"])
        steps = lut.shape[0]
        length = height if vertical else width
        for step, rgba in enumerate(lut):
            a = length * step / steps
            b = length * (step + 1) / steps
            color = imgui.get_color_u32((*rgba[:3], 1.0)) if rgba[3] > 0 else hidden
            if vertical:
                draw_list.add_rect_filled(
                    (x0, y0 + height - b), (x0 + width, y0 + height - a), color
                )
            else:
                draw_list.add_rect_filled((x0 + a, y0), (x0 + b, y0 + height), color)
        border = imgui.get_color_u32(THEME["text"])
        draw_list.add_rect((x0, y0), (x0 + width, y0 + height), border)
        text = imgui.get_color_u32(THEME["secondary"])
        marks = [
            (self.vmax, f"{self.vmax:.2f}"),
            (self.threshold, f"{self.threshold:.2f}"),
        ]
        if self.sign != "positive":
            marks += [(-self.threshold, f"{-self.threshold:.2f}")]
            marks += [(-self.vmax, f"{-self.vmax:.2f}")]
        if self.sign == "negative":
            marks = marks[2:]
        for value, label in marks:
            fraction = (value + self.vmax) / (2.0 * self.vmax)
            label_size = imgui.calc_text_size(label)
            if vertical:
                y = y0 + height * (1.0 - fraction)
                draw_list.add_line((x0 - 3, y), (x0, y), border)
                draw_list.add_text(
                    (x0 - label_size.x - 6, y - label_size.y * 0.5), text, label
                )
            else:
                x = x0 + width * fraction
                draw_list.add_line((x, y0 + height), (x, y0 + height + 3), border)
                draw_list.add_text(
                    (
                        min(max(x - label_size.x * 0.5, x0), x0 + width - label_size.x),
                        y0 + height + 4,
                    ),
                    text,
                    label,
                )
        if advance:
            imgui.dummy((width, height + imgui.get_text_line_height() + 6))


if not has_fury_v2:
    create_stat_map_visualization = StatMap3D = fury
