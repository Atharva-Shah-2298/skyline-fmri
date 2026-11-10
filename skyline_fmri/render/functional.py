"""4D BOLD runs with time navigation, summary maps and task events."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from skyline_fmri.UI.elements import (
    create_numeric_input,
    segmented_switch,
    thin_slider,
    uploader,
)
from skyline_fmri.UI.theme import CONDITION_COLORS, THEME
from skyline_fmri.fmri import (
    epi_mask,
    load_events,
    percent_change,
    temporal_snr,
    voxel_from_world,
)
from skyline_fmri.render.image import Image3D
from skyline_fmri.render.renderer import format_affine_info
from skyline_fmri.render.stat_map import StatMap3D, stat_lut

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
    from fury.lib import gfx

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui
    icons_fontawesome_6 = imgui_bundle.icons_fontawesome_6

#: "Change" shows drift-corrected percent signal change as a hot/cold
#: overlay, which is where activity is visible while the run plays.
DISPLAY_MODES = ("Volume", "Change", "Mean", "tSNR")
CHANGE_COLORMAP = "Cold-Hot"


def create_functional_visualization(
    input,
    idx,
    *,
    tr=None,
    events=None,
    render_callback=None,
    sync_callabck=None,
    time_callback=None,
    analysis_callback=None,
):
    """Create a Functional4D layer from an already-loaded BOLD run.

    Parameters
    ----------
    input : tuple
        ``(data, affine)``, ``(data, affine, filename)`` or
        ``(data, affine, filename, tr)`` where ``data`` is a 4D ndarray or
        nibabel image and ``tr`` the repetition time in seconds (or None).
    idx : int
        Index used to name the run when no filename is given.
    tr : float, optional
        Repetition time in seconds overriding the one in ``input``.
    events : dict, optional
        Task conditions as returned by :func:`skyline_fmri.fmri.load_events`.
    render_callback : callable, optional
        Callback used to request a render/update.
    sync_callabck : callable, optional
        Callback used to synchronize slice positions across visualizations.
    time_callback : callable, optional
        Called as ``time_callback(viz, seconds)`` when the user moves this
        run in time.
    analysis_callback : callable, optional
        Called as ``analysis_callback(viz, "glm")`` when the user asks for a
        task GLM.

    Returns
    -------
    Functional4D
        The created layer.

    Raises
    ------
    ValueError
        If the input is not a tuple of length 2 to 4.
    """
    if not isinstance(input, tuple) or len(input) not in (2, 3, 4):
        raise ValueError(
            "Input must be a tuple containing (data, affine), (data, affine, "
            "filename) or (data, affine, filename, tr) for BOLD visualization."
        )
    data, affine, *rest = input
    filename = rest[0] if rest else f"BOLD_{idx}"
    file_tr = rest[1] if len(rest) > 1 else None
    return Functional4D(
        filename,
        data,
        affine=affine,
        tr=tr if tr is not None else file_tr,
        events=events,
        render_callback=render_callback,
        sync_callabck=sync_callabck,
        time_callback=time_callback,
        analysis_callback=analysis_callback,
    )


class Functional4D(Image3D):
    """A BOLD run shown one volume at a time, with summary and task views.

    The slicer displays the current volume, the temporal mean, or the
    temporal SNR. Volumes are swapped in place on the GPU so scrubbing and
    playback stay smooth. The time series behind any world coordinate,
    attached task events, and a lazily computed brain mask feed the viewer's
    time-series panel and analyses.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    volume : nibabel.spatialimages.SpatialImage or ndarray
        4D run with shape ``(X, Y, Z, T)``.
    affine : ndarray, optional
        Voxel-to-world affine; uses the image affine when omitted.
    tr : float, optional
        Repetition time in seconds; 1 s with a warning when unknown.
    events : dict, optional
        Task conditions as returned by :func:`skyline_fmri.fmri.load_events`.
    opacity : int, optional
        Slice opacity in percent.
    colormap : str, optional
        Colormap for the displayed volume.
    interpolation : str, optional
        Slice interpolation (``"linear"`` or ``"nearest"``).
    render_callback : callable, optional
        Callback used to request a render/update.
    sync_callabck : callable, optional
        Callback used to synchronize slice positions across visualizations.
    time_callback : callable, optional
        Called as ``time_callback(viz, seconds)`` when the user moves this run
        in time.
    analysis_callback : callable, optional
        Called as ``analysis_callback(viz, "glm")`` when the user asks for a
        task GLM.
    """

    def __init__(
        self,
        name,
        volume,
        *,
        affine=None,
        tr=None,
        events=None,
        opacity=100,
        colormap="Gray",
        interpolation="linear",
        render_callback=None,
        sync_callabck=None,
        time_callback=None,
        analysis_callback=None,
    ):
        """Initialize the BOLD layer.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        volume : nibabel.spatialimages.SpatialImage or ndarray
            4D run with shape ``(X, Y, Z, T)``.
        affine : ndarray, optional
            Voxel-to-world affine; uses the image affine when omitted.
        tr : float, optional
            Repetition time in seconds; 1 s with a warning when unknown.
        events : dict, optional
            Task conditions as returned by
            :func:`skyline_fmri.fmri.load_events`.
        opacity : int, optional
            Slice opacity in percent.
        colormap : str, optional
            Colormap for the displayed volume.
        interpolation : str, optional
            Slice interpolation (``"linear"`` or ``"nearest"``).
        render_callback : callable, optional
            Callback used to request a render/update.
        sync_callabck : callable, optional
            Callback used to synchronize slice positions across visualizations.
        time_callback : callable, optional
            Called with ``(viz, seconds)`` when the user moves this run in time.
        analysis_callback : callable, optional
            Called with ``(viz, "glm")`` when the user asks for a task GLM.

        Raises
        ------
        ValueError
            If the run is not 4D.
        """
        if isinstance(volume, nib.spatialimages.SpatialImage):
            affine = volume.affine if affine is None else affine
            bold = volume.get_fdata(dtype=np.float32)
        else:
            bold = np.asarray(volume, dtype=np.float32)
        if bold.ndim != 4:
            raise ValueError(f"BOLD runs must be 4D; got shape {bold.shape}.")
        self.bold = bold
        self.affine = affine
        if tr is None or not tr > 0:
            logger.warning(
                f"No repetition time found for {name}; assuming 1 s. "
                "Set it in the layer controls or with --tr."
            )
            tr = 1.0
        self.tr = float(tr)
        self.events = dict(events or {})
        self.events_source = None
        self.frame = 0
        self.display_mode = "Volume"
        # Until set, the viewer may switch the run to "Change" over anatomy.
        self.mode_chosen = False
        self._summaries = {}
        self._mask = None
        self._percent = None
        self.change_threshold = None
        self.change_max = None
        self._time_callback = time_callback
        self._analysis_callback = analysis_callback
        super().__init__(
            name,
            bold[..., 0],
            affine=affine,
            interpolation=interpolation,
            render_callback=render_callback,
            opacity=opacity,
            rgb=False,
            colormap=colormap,
            sync_callabck=sync_callabck,
        )

    @property
    def viz_type(self):
        """Sidebar group identifier for BOLD runs.

        Returns
        -------
        str
            ``"functional"``.
        """
        return "functional"

    @property
    def n_frames(self):
        """Number of volumes in the run.

        Returns
        -------
        int
            Length of the time axis.
        """
        return self.bold.shape[-1]

    @property
    def times(self):
        """Acquisition time of each volume, in seconds.

        Returns
        -------
        ndarray
            ``arange(n_frames) * tr``.
        """
        return np.arange(self.n_frames) * self.tr

    @property
    def mask(self):
        """Brain mask estimated from the temporal mean, computed on first use.

        Returns
        -------
        ndarray of bool
            Spatial mask with shape ``(X, Y, Z)``.
        """
        if self._mask is None:
            self._mask = epi_mask(self._summary("Mean"))
        return self._mask

    def _populate_info(self):
        """Build the info tooltip with shape, timing and affine.

        Returns
        -------
        str
            Multi-line description of the run.
        """
        info = (
            f"Dimensions: {self.bold.shape[:3]}\n"
            f"Volumes: {self.n_frames}\n"
            f"TR: {self.tr:g} s ({self.n_frames * self.tr:g} s total)"
        )
        if self.affine is not None:
            info += "\n" + format_affine_info(self.affine)
        return info

    def _summary(self, mode):
        """Return (and cache) the temporal mean or tSNR volume."""
        if mode not in self._summaries:
            if mode == "Mean":
                self._summaries[mode] = self.bold.mean(
                    axis=-1, dtype=np.float64
                ).astype(np.float32)
            else:
                self._summaries[mode] = temporal_snr(self.bold)
        return self._summaries[mode]

    @property
    def percent(self):
        """Smoothed, drift-corrected percent signal change, computed on first use.

        Each volume is smoothed with a 1-voxel Gaussian inside the brain so
        activity reads as blobs rather than speckle. The first computation
        also picks display limits: only the strongest 5% of in-brain changes
        are shown, saturating at the 99.5th percentile.

        Returns
        -------
        ndarray, shape (X, Y, Z, T)
            Float32 percent change, zero outside the brain mask.
        """
        if self._percent is None:
            self._percent = percent_change(
                self.bold, self.tr, mask=self.mask, smooth=1.0
            )
            if self.change_max is None:
                magnitudes = np.abs(self._percent[self.mask])
                if magnitudes.size:
                    low, high = np.percentile(magnitudes, (95, 99.5))
                else:
                    low, high = 0.4, 1.0
                self.change_max = round(float(high), 2) or 1.0
                self.change_threshold = min(round(float(low), 2), self.change_max)
        return self._percent

    @property
    def is_overlay(self):
        """Whether the run is drawn as a transparent overlay (``Change``).

        Returns
        -------
        bool
            True in ``Change`` mode, where sub-threshold voxels are see-through
            and the run is stacked above opaque images.
        """
        return self.display_mode == "Change"

    def _display_volume(self):
        """Return the 3D volume the current mode and frame display."""
        if self.display_mode == "Volume":
            return self.bold[..., self.frame]
        if self.display_mode == "Change":
            return self.percent[..., self.frame]
        return self._summary(self.display_mode)

    def set_frame(self, frame, *, notify=True):
        """Show volume ``frame`` (clipped to the run).

        Parameters
        ----------
        frame : int
            Volume index.
        notify : bool, optional
            Report the new time through ``time_callback``.
        """
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        if frame == self.frame:
            return
        self.frame = frame
        if self.display_mode in ("Volume", "Change"):
            self.dwi = self._display_volume()
            self.apply_scene_op(self._upload_active_volume)
        if notify and self._time_callback is not None:
            self._time_callback(self, frame * self.tr)

    def set_time(self, seconds):
        """Show the volume acquired closest to ``seconds``.

        Parameters
        ----------
        seconds : float
            Time from the start of the run.
        """
        self.set_frame(round(seconds / self.tr), notify=False)

    def set_display_mode(self, mode):
        """Choose what the slices show.

        Parameters
        ----------
        mode : {"Volume", "Change", "Mean", "tSNR"}
            ``"Volume"`` shows raw intensities of the current volume and
            ``"Change"`` its drift-corrected percent signal change as a
            hot/cold overlay; ``"Mean"`` and ``"tSNR"`` are time summaries.

        Raises
        ------
        ValueError
            If ``mode`` is not one of :data:`DISPLAY_MODES`.
        """
        if mode not in DISPLAY_MODES:
            raise ValueError(f"mode must be one of {DISPLAY_MODES}, got {mode!r}.")
        self.display_mode = mode
        self.dwi = self._display_volume()
        self.apply_scene_op(self._show_display_volume)

    def set_change_limits(self, *, threshold=None, vmax=None):
        """Set the ``Change`` overlay's transparency threshold and saturation.

        Parameters
        ----------
        threshold : float, optional
            Smallest shown magnitude, in percent.
        vmax : float, optional
            Magnitude at which colors saturate, in percent.
        """
        if threshold is not None:
            self.change_threshold = float(threshold)
        if vmax is not None:
            self.change_max = max(float(vmax), 1e-6)
        if self.is_overlay:
            self.apply_scene_op(self._apply_colormap, self.colormap)

    def _apply_colormap(self, colormap):
        """Apply the grayscale-style colormap, or the overlay LUT in Change mode.

        Parameters
        ----------
        colormap : str
            Colormap used by the non-overlay modes.
        """
        if not self.is_overlay:
            super()._apply_colormap(colormap)
            # Opacity is only defined once the first slicer exists.
            self._set_opacity(getattr(self, "opacity", 100))
            self._slicer.render_order = 0
            return
        self.colormap = colormap
        self.percent  # noqa: B018 - computes the overlay limits on first use
        lut = stat_lut(CHANGE_COLORMAP, self.change_threshold, self.change_max, "both")
        texture_map = gfx.cm.create_colormap(lut, n=lut.shape[0])
        for actor in self._slicer.children:
            actor.material.map = texture_map
            actor.material.clim = (-self.change_max, self.change_max)
            actor.material.alpha_mode = "blend"
            actor.material.depth_write = False
        self._slicer.render_order = 10

    def draw_colorbar(self, draw_list, **kwargs):
        """Draw the ``Change`` color scale; see ``StatMap3D.draw_colorbar``.

        Parameters
        ----------
        draw_list : ImDrawList
            Target draw list.
        **kwargs : dict
            ``origin``, ``size`` and ``vertical``, as for statistical maps.
        """
        scale = SimpleNamespace(
            colormap=CHANGE_COLORMAP,
            threshold=self.change_threshold,
            vmax=self.change_max,
            sign="both",
        )
        StatMap3D.draw_colorbar(scale, draw_list, **kwargs)

    def _show_display_volume(self):
        """Upload the displayed volume and refit intensity limits to it."""
        self._upload_active_volume()
        self._apply_colormap(self.colormap)

    def set_tr(self, tr):
        """Change the repetition time.

        Parameters
        ----------
        tr : float
            Repetition time in seconds; ignored unless positive.
        """
        if tr > 0:
            self.tr = float(tr)
            self._info = self._populate_info()
            if self._percent is not None:
                # The drift model depends on the TR.
                self._percent = None
                if self.display_mode == "Change":
                    self.set_display_mode("Change")

    def set_events(self, events, *, source=None):
        """Attach task conditions to this run.

        Parameters
        ----------
        events : dict
            Condition name to ``(N, 2)`` ``(onset, duration)`` array.
        source : str, optional
            File the events came from, shown in the controls.
        """
        self.events = dict(events)
        self.events_source = source

    def load_events_file(self, fname):
        """Read and attach a BIDS events file, logging failures.

        Parameters
        ----------
        fname : str
            Path to a ``.tsv`` or ``.csv`` events table.

        Returns
        -------
        bool
            Whether events were attached.
        """
        try:
            events = load_events(fname)
        except (OSError, ValueError) as e:
            logger.error(f"Could not read events from {fname}: {e}")
            return False
        self.set_events(events, source=fname)
        return True

    def timeseries_at(self, world):
        """Return the voxel and time series under a world coordinate.

        Parameters
        ----------
        world : array-like, shape (3,)
            RAS+ coordinate in millimeters.

        Returns
        -------
        tuple or None
            ``(voxel_index, series)`` or None when the point is outside the
            run's field of view.
        """
        voxel = voxel_from_world(world, self.affine, self.bold.shape)
        if voxel is None:
            return None
        return voxel, self.bold[voxel]

    def render_widgets(self):
        """Draw slice, time, display, timing, task and voxel controls."""
        self._render_sync_controls()
        self._render_opacity_control()
        imgui.spacing()
        self._render_slice_controls()
        imgui.spacing()
        self._render_time_controls()
        imgui.spacing()
        if self.is_overlay:
            self._render_change_controls()
        else:
            self._render_intensity_control()
            imgui.spacing()
            self._render_colormap_control()
        imgui.spacing()
        self._render_task_controls()
        imgui.spacing()
        self._render_voxel_info()
        imgui.spacing()
        imgui.spacing()
        self._render_interpolation_control()
        imgui.spacing()

    def _render_time_controls(self):
        """Draw the volume slider, display mode switch and TR input."""
        changed, value = thin_slider(
            "Volume",
            self.frame,
            0,
            self.n_frames - 1,
            value_type="int",
            text_format=".0f",
            step=1,
        )
        if changed:
            self.set_frame(int(value))
        imgui.text_colored(
            THEME["text"],
            f"t = {self.frame * self.tr:.1f} s  of  {self.n_frames * self.tr:.0f} s",
        )
        imgui.spacing()
        # segmented_switch reports title-cased options ("tSNR" -> "Tsnr").
        modes = {mode.title(): mode for mode in DISPLAY_MODES}
        changed, mode = segmented_switch(
            "Show", list(DISPLAY_MODES), self.display_mode.title()
        )
        if changed:
            self.mode_chosen = True
            self.set_display_mode(modes[mode])
        imgui.spacing()
        changed, value = create_numeric_input(
            "TR (s)", self.tr, value_type="float", step=0.1, format="%.3f"
        )
        if changed:
            self.set_tr(value)

    def _render_change_controls(self):
        """Draw the Change threshold and saturation sliders and color scale."""
        top = max(self.change_max * 3, 1.0)
        changed, value = thin_slider(
            "Show above",
            self.change_threshold,
            0.0,
            top,
            text_format=".2f",
            value_unit=" %",
            step=top / 200.0,
        )
        if changed:
            self.set_change_limits(threshold=min(value, self.change_max))
        changed, value = thin_slider(
            "Max",
            self.change_max,
            0.01,
            top,
            text_format=".2f",
            value_unit=" %",
            step=top / 200.0,
        )
        if changed:
            self.set_change_limits(vmax=max(value, self.change_threshold + 1e-3))
        imgui.spacing()
        self.draw_colorbar(imgui.get_window_draw_list())

    def _render_task_controls(self):
        """Draw the attached conditions, the events loader and the GLM action."""
        icon = icons_fontawesome_6.ICON_FA_LIST_CHECK
        imgui.text_colored(THEME["text"], f"{icon}  Task events")
        for index, (condition, rows) in enumerate(self.events.items()):
            color = CONDITION_COLORS[index % len(CONDITION_COLORS)]
            imgui.text_colored(color, icons_fontawesome_6.ICON_FA_SQUARE)
            imgui.same_line(0, 6)
            imgui.text_colored(THEME["secondary"], f"{condition}  ({len(rows)})")
        uploader(
            "Load events (.tsv)",
            self._on_events_selected,
            extension="*.tsv *.csv",
            type="viz",
        )
        if self.events and self._analysis_callback is not None:
            imgui.spacing()
            glm_icon = icons_fontawesome_6.ICON_FA_CALCULATOR
            imgui.text_colored(THEME["primary"], f"{glm_icon}  Fit task GLM")
            if imgui.is_item_hovered():
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
                imgui.set_tooltip(
                    "Fit the HRF-convolved conditions with a cosine high-pass "
                    "and add one t-map per condition."
                )
            if imgui.is_item_clicked():
                self._analysis_callback(self, "glm")

    def _on_events_selected(self, *, filenames=None, **_):
        """Attach the events file chosen in the uploader dialog."""
        if filenames:
            self.load_events_file(filenames[0])


if not has_fury_v2:
    create_functional_visualization = Functional4D = fury
