"""Skyline viewer application entry points.

Expose the ``Skyline`` class and the ``skyline``/``skyline_from_files``
functions used to construct and launch the FURY-based multi-modal viewer.
"""

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import time

import numpy as np

from dipy.io.utils import split_filename_extension
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from skyline_fmri.UI.manager import UIWindow
from skyline_fmri.UI.theme import LOGO_SMALL, THEME
from skyline_fmri._compat import gizmo
from skyline_fmri.compute import process_async_callbacks, run_async
from skyline_fmri.fmri import (
    fit_glm,
    load_events,
    seed_correlation,
    voxel_from_world,
)
from skyline_fmri.io import _valid_shm_coeffs, _validate_sh_basis, load_files
from skyline_fmri.tracts import (
    TractIndex,
    streamlines_through,
    tract_key,
    tract_label,
)

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
    from fury.actor import Actor, show_slices
    from fury.colormap import distinguishable_colormap
    from fury.io import load_image_as_wgpu_texture_view
    from fury.lib import gfx
    from fury.window import calculate_screen_sizes, update_camera, update_viewports

    from skyline_fmri.UI.timeseries import TimeseriesPanel
    from skyline_fmri.render.functional import (
        Functional4D,
        create_functional_visualization,
    )
    from skyline_fmri.render.image import Image3D, create_image_visualization
    from skyline_fmri.render.peak import Peak3D, create_peak_visualization
    from skyline_fmri.render.renderer import (
        affine_voxel_sizes,
        create_window,
        slice_slider_bounds,
        slice_slider_values_from_state,
        slice_state_from_slider_values,
    )
    from skyline_fmri.render.roi import ROI3D, create_roi_visualization
    from skyline_fmri.render.sh_slicer import SHGlyph3D, create_shm_visualization
    from skyline_fmri.render.stat_map import (
        StatMap3D,
        create_stat_map_visualization,
    )
    from skyline_fmri.render.streamline import (
        ClusterStreamline3D,
        Streamline3D,
        create_cluster_help,
        create_streamline_visualization,
    )
    from skyline_fmri.render.surface import Surface, create_surface_visualization

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui

# Gap between coplanar slices, as a fraction of the camera distance.
_DEPTH_GAP = 5e-4

#: View layouts: one 3D view, or 2D axial/sagittal/coronal views plus 3D.
LAYOUTS = ("3d", "triplanar")
#: Backgrounds offered by :meth:`Skyline.capture_figure`.
FIGURE_BACKGROUNDS = ("scene", "white", "black", "transparent")

# Side each 2D camera looks from (left, posterior, superior).
_PLANAR_SIDES = np.array([-1.0, -1.0, 1.0])
# Name, viewed axis, camera up, and the (top, bottom, left, right) letters.
_PLANAR_VIEWS = (
    ("Axial", 2, (0.0, 1.0, 0.0), ("A", "P", "L", "R")),
    ("Sagittal", 0, (0.0, 0.0, 1.0), ("S", "I", "A", "P")),
    ("Coronal", 1, (0.0, 0.0, 1.0), ("S", "I", "L", "R")),
)

# Half thickness (mm) of the tract/surface/ROI slab shown in 2D views.
PLANAR_SLAB_MM = 5.0
_NULL_PLANES = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]


if has_fury_v2:

    class _SlabViewport(gfx.Viewport):
        """Viewport that clips 3D geometry to a slab around one slice plane.

        Parameters
        ----------
        renderer : Renderer
            Renderer the viewport draws into.
        viewer : Skyline
            Viewer providing the clipped materials and the slice position.
        axis : int
            World axis normal to the slice shown in this viewport.
        """

        def __init__(self, renderer, viewer, axis):
            """Create the viewport for the slice normal to ``axis``."""
            super().__init__(renderer)
            self._viewer = viewer
            self._axis = axis

        @contextmanager
        def slab(self):
            """Clip tract, surface and ROI geometry to this view's slab.

            pygfx materials keep fragments with ``a x + b y + c z >= d``.
            Geometry whose material ignores clipping planes (FURY's baked
            streamtubes) is hidden instead. Each render call submits its own
            GPU work, so these changes only affect renders made inside the
            block; they are undone afterwards so the 3D view, rendered first,
            never sees them.
            """
            clipped, hidden = self._viewer._slab_objects()
            if clipped:
                position = self._viewer._slice_position()[self._axis]
                normal = np.zeros(3)
                normal[self._axis] = 1.0
                planes = [
                    (*normal, position - PLANAR_SLAB_MM),
                    (*(-normal), -(position + PLANAR_SLAB_MM)),
                ]
                for obj in clipped:
                    obj.material.clipping_planes = planes
            for obj in hidden:
                obj.visible = False
            try:
                yield
            finally:
                for obj in clipped:
                    obj.material.clipping_planes = _NULL_PLANES
                for obj in hidden:
                    obj.visible = True

        def render(self, scene, camera, flush=False):  # pep3102: ignore
            """Render with 3D geometry clipped to the slab around the slice.

            The shared scene's UI layer (the orientation gizmo) belongs to the
            3D view; the 2D views show direction letters instead.
            """
            if scene is self._viewer.window.screens[0].scene.ui_scene:
                return
            with self.slab():
                super().render(scene, camera, flush=flush)


class Skyline:
    """The Skyline viewer, hosting the FURY scene, UI, and visualizations.

    Parameters
    ----------
    visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
        Kind of window to create. The options map to FURY window types via
        ``create_window``:

        - "standalone": a default interactive window.
        - "gui": a Qt-based window.
        - "jupyter": an inline Jupyter notebook window.
        - "stealth": an offscreen window with no GUI, used for scripted
          snapshots.

        An unrecognized value logs an error and terminates the process.
        ``DIPY_OFFSCREEN=1`` or ``true`` (case-insensitive) forces stealth.
        Otherwise, ``visualizer_type`` controls rendering.
    images : list of tuple, optional
        ``(data, affine)`` or ``(data, affine, filename)`` tuples where
        ``data`` is a nibabel image or ndarray and ``filename`` is a display label.
    peaks : list of tuple, optional
        (peak_dirs, affine, filename, peak_values) tuples; see
        `create_peak_visualization`.
    rois : list of tuple, optional
        Already-loaded ROI data to show at startup, as ``(roi, affine)`` or
        ``(roi, affine, filename)`` tuples.
    surfaces : list of tuple, optional
        Already-loaded surface data to show at startup, as
        ``(vertices, faces)`` or ``(vertices, faces, filename)`` tuples.
    tractograms : list of tuple, optional
        Already-loaded tractogram data to show at startup, as ``(sft,)`` or
        ``(sft, filename)`` tuples, where ``sft`` is a
        ``StatefulTractogram``. Entries with no streamlines are skipped
        with a warning.
    sh_coeffs : list of tuple, optional
        Already-loaded spherical harmonic coefficient data to show at
        startup, as ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
        ``(coeffs, affine, filename, basis_type)`` tuples. ``coeffs`` must
        be a 4D ndarray with a symmetric SH count, or a full SH count when the
        fourth tuple element is ``"standard"``. Invalid entries are logged
        and skipped.
    functional : list of tuple, optional
        Already-loaded 4D BOLD runs as ``(data, affine)``,
        ``(data, affine, filename)`` or ``(data, affine, filename, tr)``
        tuples, where ``data`` is a nibabel image or ndarray and ``tr`` the
        repetition time in seconds (or None).
    stat_maps : list of tuple, optional
        Already-loaded 3D statistical maps (t, z, correlation, ...) shown as
        thresholded activation overlays, as ``(data, affine)`` or
        ``(data, affine, filename)`` tuples.
    events : str, list of str or dict, optional
        BIDS events table (``.tsv``/``.csv``; the first path of a list), or a
        mapping of condition name to ``(N, 2)`` ``(onset, duration)`` arrays
        in seconds, attached to every BOLD run.
    tr : float, optional
        Repetition time in seconds for every BOLD run, overriding the BIDS
        sidecar and NIfTI header.
    sh_basis : str, optional
        Input convention for ODF files and unlabelled arrays: ``"descoteaux07"``
        (latest), ``"descoteaux07_legacy"`` (old DIPY), ``"tournier07"``
        (MRtrix 0.2), or ``"tournier19"`` (MRtrix3). A fourth tuple element
        overrides this selection.
        Unsupported values are logged and use latest Descoteaux07.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
        which improves performance for large tractograms.
    glass_brain : bool, optional
        Whether to render surfaces black with the ``"basic"`` material at
        25% opacity and default the background to white.
    bg_color : tuple of float, optional
        Background color of the scene as an RGB tuple in ``[0, 1]``. If
        None, it is white when ``glass_brain`` is True, otherwise dark
        gray.
    tract_colors : str or tuple of float or None, optional
        Coloring scheme for the tractograms: ``"direction"`` for
        directionally colored streamlines, ``"random"`` for the next color
        from a distinguishable colormap per tractogram, an RGB(A) tuple in
        ``[0, 1]``, or a string of three space-separated numbers parsed to
        such a tuple. If None, ``"direction"`` is used.
    cluster_thr : float, optional
        Final distance threshold, in mm, used by ``qbx_and_merge`` when
        clustering is enabled; small-animal data may need a smaller value
        such as 2.0.
    cluster_size_thr : int, optional
        Clusters with size less than ``cluster_size_thr`` are hidden. If
        None, the 50th percentile of the cluster size distribution is
        used.
    cluster_length_thr : float, optional
        Clusters with average length less than ``cluster_length_thr`` mm
        are hidden. If None, the 25th percentile of the cluster length
        distribution is used.
    buan_pvals : str, optional
        File path for BUAN p-values used for BUAN-based coloring of
        tractograms.
    rgb : bool or None, optional
        ``None``: auto-detect from structured NIfTI ``DT_RGB24``
        dtype; show toggle for other 4D volumes with 3 or 4 channels.
        ``True``: force RGB mode.  ``False``: never treat as RGB.
    initial_filenames : list of str, optional
        File paths loaded asynchronously into the viewer on startup. If
        neither preloaded data nor initial files are given and a UI
        exists, the file dialog opens on start.
    initial_rois : list of str, optional
        ROI file paths loaded asynchronously into the viewer on startup.
    initial_peaks : list of str, optional
        List of ``.pam5`` or NIfTI (.nii, .nii.gz) peak file paths to load
        into the Skyline viewer on startup.
    initial_shm_coeffs : list of str, optional
        Spherical harmonic coefficient file paths loaded asynchronously
        into the viewer on startup.
    initial_functional : list of str, optional
        4D BOLD file paths loaded asynchronously into the viewer on startup.
    initial_stat_maps : list of str, optional
        Statistical map file paths loaded asynchronously into the viewer on
        startup.
    layout : {"3d", "triplanar"}, optional
        Initial view layout: the 3D view, or 2D axial, sagittal and coronal
        views next to the 3D view. The sidebar button switches it later.
    out_dir : str or Path, optional
        Directory for the offscreen output image; created if missing.
    out_stealth_png : str, optional
        Output image name used as the offscreen window title. Applies when
        stealth mode is requested or ``DIPY_OFFSCREEN`` is ``1``/``true``.

    Notes
    -----
    Window creation sets ``FURY_OFFSCREEN`` to ``1`` for stealth or removes
    the key otherwise. Skyline restores its previous value or absence after
    setup/start returns or raises.
    """

    def __init__(
        self,
        *,
        visualizer_type="standalone",
        images=None,
        peaks=None,
        rois=None,
        surfaces=None,
        tractograms=None,
        sh_coeffs=None,
        functional=None,
        stat_maps=None,
        events=None,
        tr=None,
        sh_basis="descoteaux07",
        is_cluster=False,
        is_light_version=False,
        glass_brain=False,
        bg_color=None,
        tract_colors=None,
        cluster_thr=15.0,
        cluster_size_thr=None,
        cluster_length_thr=None,
        buan_pvals=None,
        rgb=None,
        initial_filenames=None,
        initial_rois=None,
        initial_peaks=None,
        initial_shm_coeffs=None,
        initial_functional=None,
        initial_stat_maps=None,
        layout="3d",
        out_dir=None,
        out_stealth_png=None,
    ):
        """Initialize the Skyline viewer.

        Blocks in ``self.window.start()`` until the window is closed.

        Parameters
        ----------
        visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
            Kind of window to create. The options map to FURY window types via
            ``create_window``:

            - "standalone": a default interactive window.
            - "gui": a Qt-based window.
            - "jupyter": an inline Jupyter notebook window.
            - "stealth": an offscreen window with no GUI, used for scripted
              snapshots.

            An unrecognized value logs an error and terminates the process.
            ``DIPY_OFFSCREEN=1`` or ``true`` (case-insensitive) forces stealth.
            Otherwise, ``visualizer_type`` controls rendering.
        images : list of tuple, optional
            ``(data, affine)`` or ``(data, affine, filename)`` tuples where
            ``data`` is a nibabel image or ndarray and ``filename`` a display label.
        peaks : list of tuple, optional
            (peak_dirs, affine, filename, peak_values) tuples; see
            `create_peak_visualization`.
        rois : list of tuple, optional
            Already-loaded ROI data to show at startup, as ``(roi, affine)`` or
            ``(roi, affine, filename)`` tuples.
        surfaces : list of tuple, optional
            Already-loaded surface data to show at startup, as
            ``(vertices, faces)`` or ``(vertices, faces, filename)`` tuples.
        tractograms : list of tuple, optional
            Already-loaded tractogram data to show at startup, as ``(sft,)`` or
            ``(sft, filename)`` tuples, where ``sft`` is a
            ``StatefulTractogram``. Entries with no streamlines are skipped
            with a warning.
        sh_coeffs : list of tuple, optional
            Already-loaded spherical harmonic coefficient data to show at
            startup, as ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
            ``(coeffs, affine, filename, basis_type)`` tuples. ``coeffs`` must
            be a 4D ndarray with a symmetric SH count, or a full SH count when the
            fourth tuple element is ``"standard"``. Invalid entries are logged
            and skipped.
        functional : list of tuple, optional
            Already-loaded 4D BOLD runs as ``(data, affine)``,
            ``(data, affine, filename)`` or ``(data, affine, filename, tr)``
            tuples, where ``data`` is a nibabel image or ndarray and ``tr`` the
            repetition time in seconds (or None).
        stat_maps : list of tuple, optional
            Already-loaded 3D statistical maps (t, z, correlation, ...) shown
            as thresholded activation overlays, as ``(data, affine)`` or
            ``(data, affine, filename)`` tuples.
        events : str, list of str or dict, optional
            BIDS events table (``.tsv``/``.csv``; the first path of a list), or
            a mapping of condition name to ``(N, 2)`` ``(onset, duration)``
            arrays in seconds, attached to every BOLD run.
        tr : float, optional
            Repetition time in seconds for every BOLD run, overriding the BIDS
            sidecar and NIfTI header.
        sh_basis : str, optional
            Input convention for ODF files and unlabelled arrays: ``"descoteaux07"``
            (latest), ``"descoteaux07_legacy"`` (old DIPY), ``"tournier07"``
            (MRtrix 0.2), or ``"tournier19"`` (MRtrix3). A fourth tuple element
            overrides this selection.
            Unsupported values are logged and use latest Descoteaux07.
        is_cluster : bool, optional
            Whether to cluster the tractograms.
        is_light_version : bool, optional
            Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
            which improves performance for large tractograms.
        glass_brain : bool, optional
            Whether to render surfaces black with the ``"basic"`` material at
            25% opacity and default the background to white.
        bg_color : tuple of float, optional
            Background color of the scene as an RGB tuple in ``[0, 1]``. If
            None, it is white when ``glass_brain`` is True, otherwise dark
            gray.
        tract_colors : str or tuple of float or None, optional
            Coloring scheme for the tractograms: ``"direction"`` for
            directionally colored streamlines, ``"random"`` for the next color
            from a distinguishable colormap per tractogram, an RGB(A) tuple in
            ``[0, 1]``, or a string of three space-separated numbers parsed to
            such a tuple. If None, ``"direction"`` is used.
        cluster_thr : float, optional
            Final distance threshold, in mm, used by ``qbx_and_merge`` when
            clustering is enabled; small-animal data may need a smaller value
            such as 2.0.
        cluster_size_thr : int, optional
            Clusters with size less than ``cluster_size_thr`` are hidden. If
            None, the 50th percentile of the cluster size distribution is
            used.
        cluster_length_thr : float, optional
            Clusters with average length less than ``cluster_length_thr`` mm
            are hidden. If None, the 25th percentile of the cluster length
            distribution is used.
        buan_pvals : str, optional
            File path for BUAN p-values used for BUAN-based coloring of
            tractograms.
        rgb : bool or None, optional
            ``None``: auto-detect from structured NIfTI ``DT_RGB24``
            dtype; show toggle for other 4D volumes with 3 or 4 channels.
            ``True``: force RGB mode.  ``False``: never treat as RGB.
        initial_filenames : list of str, optional
            File paths loaded asynchronously into the viewer on startup. If
            neither preloaded data nor initial files are given and a UI
            exists, the file dialog opens on start.
        initial_rois : list of str, optional
            ROI file paths loaded asynchronously into the viewer on startup.
        initial_peaks : list of str, optional
            List of ``.pam5`` or NIfTI (.nii, .nii.gz) peak file paths to load
            into the Skyline viewer on startup.
        initial_shm_coeffs : list of str, optional
            Spherical harmonic coefficient file paths loaded asynchronously
            into the viewer on startup.
        initial_functional : list of str, optional
            4D BOLD file paths loaded asynchronously into the viewer on startup.
        initial_stat_maps : list of str, optional
            Statistical map file paths loaded asynchronously into the viewer on
            startup.
        layout : {"3d", "triplanar"}, optional
            Initial view layout: the 3D view, or 2D axial, sagittal and coronal
            views next to the 3D view.
        out_dir : str or Path, optional
            Directory for the offscreen output image; created if missing.
        out_stealth_png : str, optional
            Output image name used as the offscreen window title. Applies when
            stealth mode is requested or ``DIPY_OFFSCREEN`` is ``1``/``true``.

        Notes
        -----
        Window creation sets ``FURY_OFFSCREEN`` to ``1`` for stealth or removes
        the key otherwise. Skyline restores its previous value or absence after
        setup/start returns or raises.
        """
        sh_basis = _validate_sh_basis(sh_basis)
        self.size = (1200, 1000)
        self.ui_size = (400, self.size[1])
        dipy_offscreen_requested = os.environ.get("DIPY_OFFSCREEN", "").lower() in (
            "1",
            "true",
        )
        self._visualizer_type = (
            "stealth" if dipy_offscreen_requested else visualizer_type
        )
        self._direct_load = True
        self._rgb = rgb
        self._cluster_thr = cluster_thr
        self._cluster_size_thr = cluster_size_thr
        self._cluster_length_thr = cluster_length_thr
        self._buan_pvals = buan_pvals
        self._sh_basis = sh_basis
        self._tr = tr
        self._events = self._resolve_events(events)
        self._cursor = None
        self._crosshair = None
        self._time = 0.0
        self._dock_height = 0
        self._timeseries_panel = None
        self._was_playing = False
        self._layout = "3d"
        self._planar_views = []
        self._pointer = None
        self._isolated_tracts = set()
        self._tract_index = None
        self._cursor_info_key = None
        self._cursor_info = []
        previous_fury_offscreen = os.environ.get("FURY_OFFSCREEN")
        try:
            if self._visualizer_type != "stealth":
                self.window = create_window(
                    visualizer_type=self._visualizer_type,
                    size=self.size,
                    screen_config=[
                        (
                            self.ui_size[0],
                            0,
                            self.size[0] - self.ui_size[0],
                            self.size[1],
                        ),
                    ],
                )
                gizmo(self.window)
            else:
                title = "SKYLINE fMRI"
                if out_stealth_png is not None:
                    title = split_filename_extension(out_stealth_png)[0]
                if out_dir is not None and out_dir != "":
                    os.makedirs(out_dir, exist_ok=True)
                    title = os.path.join(out_dir, title)
                self.window = create_window(
                    visualizer_type=self._visualizer_type, size=self.size, title=title
                )
            if bg_color is None:
                bg_color = (1, 1, 1) if glass_brain else (0.1, 0.1, 0.1)
            self._bg_color = bg_color
            self.window.screens[0].scene.background = self._bg_color

            if tract_colors is None:
                tract_colors = "direction"
            elif isinstance(tract_colors, str) and len(tract_colors.split(" ")) == 3:
                tract_colors = tuple(map(float, tract_colors.split(" ")))
            self._tract_colors = tract_colors

            self._image_visualizations = []
            self._peak_visualizations = []
            self._roi_visualizations = []
            self._surface_visualizations = []
            self._tractogram_visualizations = []
            self._sh_glyph_visualizations = []
            self._pending_loaded_files = []
            self._pending_tractogram_switches = []
            self._loading_total = 0
            self._loading_done = 0
            self._is_drawing_ui = False
            self._refresh_requested = False
            self._pending_bg_color = None
            self._pending_sync_requests = []
            self._pending_scene_ops = []
            self._is_cluster = is_cluster
            self._is_light_version = is_light_version
            self._glass_brain = glass_brain
            self._tractogram_help = False
            self.window.renderer.add_event_handler(self.handle_key_events, "key_down")
            self.window.renderer.add_event_handler(self._track_pointer, "pointer_down")
            self.window.resize_callback(self.handle_resize)
            self._color_gen = distinguishable_colormap()
            self.active_image = None
            self._slice_focus_viz = None

            if self._visualizer_type != "stealth":
                gpu_texture = load_image_as_wgpu_texture_view(
                    str(LOGO_SMALL), self.window.device
                )
                logo_tex_ref = self.window._imgui.backend.register_texture(gpu_texture)
                self.UI_window = UIWindow(
                    "Image Controls",
                    size=self.ui_size,
                    render_callback=self.request_refresh,
                    logo_tex_ref=logo_tex_ref,
                    file_dialog_callback=self._append_visualization,
                    bg_color_callback=self._update_background_color,
                    snapshot_callback=self._save_snapshot,
                    sh_basis=sh_basis,
                    slice_controls_callback=self._get_global_slice_controls,
                    slice_position_callback=self._set_global_slice_position,
                    slice_visibility_callback=self._set_global_slice_visibility,
                    layout_callback=self._layout_ui,
                )
                self.window._imgui.set_gui(self.draw_ui)
                self._timeseries_panel = TimeseriesPanel(
                    time_callback=self.set_time,
                    seed_callback=self.compute_seed_correlation,
                )
            else:
                self.UI_window = None

            initial_loaded_files = {
                "images": images or [],
                "peaks": peaks or [],
                "rois": rois or [],
                "surfaces": surfaces or [],
                "tractograms": tractograms or [],
                "shm_coeffs": sh_coeffs or [],
                "functional": functional or [],
                "stat_maps": stat_maps or [],
            }
            has_initial_visualizations = any(initial_loaded_files.values())
            has_initial_files = any(
                (
                    initial_filenames,
                    initial_rois,
                    initial_peaks,
                    initial_shm_coeffs,
                    initial_functional,
                    initial_stat_maps,
                )
            )

            if has_initial_visualizations:
                self._queue_loaded_visualizations(initial_loaded_files)

            if has_initial_files:
                self._append_visualization(
                    filenames=initial_filenames,
                    rois=initial_rois,
                    peaks=initial_peaks,
                    shm_coeffs=initial_shm_coeffs,
                    functional=initial_functional,
                    stat_maps=initial_stat_maps,
                )
            elif not has_initial_visualizations and self.UI_window is not None:
                self.UI_window.request_file_dialog = True

            if self._visualizer_type == "stealth":
                self._wait_for_loading_in_stealth_mode()
            self.set_layout(layout)

            self.before_render()
            self._direct_load = False
            self.window.start()
        finally:
            if previous_fury_offscreen is None:
                os.environ.pop("FURY_OFFSCREEN", None)
            else:
                os.environ["FURY_OFFSCREEN"] = previous_fury_offscreen

    def _wait_for_loading_in_stealth_mode(self):
        """Block until all queued and pending visualizations finish loading.

        Repeatedly runs the async callback queue and drains pending loaded
        files so stealth-mode snapshots are not taken before every
        requested visualization has been created.
        """
        while self._pending_loaded_files or (
            self._loading_total > 0 and self._loading_done < self._loading_total
        ):
            process_async_callbacks()
            self._drain_pending_visualizations()
            time.sleep(0.01)

    def _refresh_actors(self):
        """Sync the main scene's actors with the current visualizations.

        Removes actors that no longer belong to any visualization and adds
        actors for visualizations not yet present in the scene.
        """
        all_actors = [v.actor for v in self.visualizations]

        for actor in list(self.window.screens[0].scene.main_scene.children):
            if not isinstance(actor, Actor):
                continue
            if not any(a == actor for a in all_actors):
                self.window.screens[0].scene.main_scene.remove(actor)
        for a in all_actors:
            if a not in self.window.screens[0].scene.main_scene.children:
                self.window.screens[0].scene.main_scene.add(a)

    def _refresh_ui(self):
        """Drop tracked visualizations whose UI section was closed.

        A visualization removed from the UI (for example via its close
        button) no longer has a matching section id, so it is unregistered
        from the viewer as well.
        """
        for viz in self.visualizations:
            viz_id = f"{viz.path}:{viz.name}"
            if viz_id not in self.UI_window.sections:
                self._remove_visualization(viz)

    def _arrange_image_actors(self):
        """Stack coplanar image slices toward the camera to avoid z-fighting.

        Images keep their load order, the active (selected) image is drawn
        above the other images and statistical overlays above everything.
        Each layer's planes are pushed toward the camera by a gap that grows
        with the camera distance, so the depth buffer always separates them
        and slices do not flicker while the view moves.
        """
        for viz in self._image_visualizations:
            if viz.active:
                self.active_image = viz
        overlays = [
            viz for viz in self._functional_visualizations if viz.is_overlay
        ] + self._stat_map_visualizations
        stack = [
            viz
            for viz in self._image_visualizations
            if viz is not self.active_image and viz not in overlays
        ]
        if self.active_image is not None and self.active_image not in overlays:
            stack.append(self.active_image)
        stack += overlays
        camera = np.asarray(self.window.screens[0].camera.world.position, dtype=float)
        for rank, viz in enumerate(stack):
            state = np.asarray(viz.state, dtype=float)
            step = max(np.linalg.norm(camera - state) * _DEPTH_GAP, 1e-3)
            if self._layout == "triplanar":
                side = _PLANAR_SIDES
            else:
                side = np.where(camera >= state, 1.0, -1.0)
            show_slices(viz.actor, state + side * rank * step)

    def _update_tractogram_helper(self, *, remove=False):
        """Show or hide the cluster interaction help overlay.

        The overlay is added when a ``ClusterStreamline3D`` visualization is
        present and removed once none remain, or immediately when
        ``remove`` is True.

        Parameters
        ----------
        remove : bool, optional
            Whether to force-remove the overlay regardless of its current
            visualizations.
        """
        if remove and self._tractogram_help:
            self.window.screens[0].scene.remove(self._tractogram_help)
            self._tractogram_help = False

        if (
            any(
                isinstance(viz, ClusterStreamline3D)
                for viz in self._tractogram_visualizations
            )
            and not self._tractogram_help
        ):
            self._tractogram_help = create_cluster_help(
                position=(self.size[0] - self.ui_size[0] - 200, 0)
            )
            self.window.screens[0].scene.add(self._tractogram_help)
        elif (
            not any(
                isinstance(viz, ClusterStreamline3D)
                for viz in self._tractogram_visualizations
            )
            and self._tractogram_help
        ):
            self.window.screens[0].scene.remove(self._tractogram_help)
            self._tractogram_help = False

    def draw_ui(self):
        """Draw the ImGui overlay for a single frame.

        Invoked as the ImGui GUI callback. Renders the UI window, then
        drains pending tractogram switches, visualizations, synchronization
        requests, and scene operations queued while drawing, applying a
        pending background color change and refreshing if required.
        """
        process_async_callbacks()
        self._is_drawing_ui = True
        try:
            if len(self.visualizations) == 0 and self._loading_total == 0:
                self.UI_window.request_file_dialog = True
            else:
                self.UI_window.request_file_dialog = False
            self.UI_window.render()
            self._draw_fmri_ui()
        finally:
            self._is_drawing_ui = False

        self._process_tractogram_switches()
        self._drain_pending_visualizations()
        self._flush_pending_sync_requests()
        self._flush_pending_scene_ops()
        if self._pending_bg_color is not None:
            self.window.screens[0].scene.background = self._pending_bg_color
            self._bg_color = self._pending_bg_color
            self._pending_bg_color = None
            self._refresh_requested = True
        self.active_image and self._arrange_image_actors()
        if self._refresh_requested:
            self.before_render()

    def request_refresh(self):
        """Flag the viewer for a refresh on the next UI frame.

        The actual actor sync and render happen later, either at the end of
        the current ``draw_ui`` call or on the next ``before_render`` call.
        """
        self._refresh_requested = True

    @staticmethod
    def _resolve_events(events):
        """Return task events as a condition mapping, loading files as needed.

        Parameters
        ----------
        events : str, Path, list of str, dict or None
            Events table path (the first one of a list) or an already-built
            mapping of condition to ``(N, 2)`` ``(onset, duration)`` arrays.

        Returns
        -------
        dict or None
            Condition mapping, or None when no events were given or the file
            could not be read (the failure is logged).
        """
        if events is None or isinstance(events, dict):
            return events
        if isinstance(events, (list, tuple)):
            if not events:
                return None
            events = events[0]
        try:
            return load_events(events)
        except (OSError, ValueError) as e:
            logger.error(f"Could not read events from {events}: {e}")
            return None

    @property
    def _functional_visualizations(self):
        """BOLD runs among the loaded images, in load order."""
        return [v for v in self._image_visualizations if isinstance(v, Functional4D)]

    @property
    def _stat_map_visualizations(self):
        """Statistical overlays among the loaded images, in load order."""
        return [v for v in self._image_visualizations if isinstance(v, StatMap3D)]

    @property
    def cursor(self):
        """World coordinate (RAS+ mm) of the picked voxel, or None.

        Returns
        -------
        ndarray or None
            The cursor set by clicking a slice or :meth:`set_cursor`.
        """
        return self._cursor

    @property
    def current_time(self):
        """Time, in seconds, shown by every BOLD run.

        Returns
        -------
        float
            Seconds from the start of the runs.
        """
        return self._time

    def _on_pick(self, viz, world, *, center):
        """Move the cursor to a clicked slice point.

        Parameters
        ----------
        viz : Image3D
            Image whose slice was clicked.
        world : ndarray, shape (3,)
            Clicked point in world coordinates.
        center : bool
            Also move the synchronized slices through the point.
        """
        if self._pointer is not None and self._planar_view_at(*self._pointer):
            # In a 2D view a click moves the crosshair, as in other viewers.
            center = True
        self.set_cursor(world, center=center)

    def set_cursor(self, world, *, center=False):
        """Pick the voxel under a world coordinate for time-series display.

        Parameters
        ----------
        world : array-like, shape (3,)
            RAS+ coordinate in millimeters.
        center : bool, optional
            Also move every synchronized image, peak and ODF slice through
            the point.
        """
        self._cursor = np.asarray(world, dtype=float)[:3].copy()
        if center:
            for viz in (
                self._image_visualizations
                + self._peak_visualizations
                + self._sh_glyph_visualizations
            ):
                viz.update_state(self._cursor)
        self.enqueue_scene_op(self._update_crosshair)

    def _update_crosshair(self):
        """Draw a small three-axis cross at the cursor, above all slices."""
        if self._cursor is None:
            return
        voxel_sizes = [
            affine_voxel_sizes(viz.affine).min()
            for viz in self._image_visualizations
            if viz.affine is not None
        ]
        arm = 4.0 * (min(voxel_sizes) if voxel_sizes else 1.0)
        offsets = np.repeat(np.eye(3) * arm, 2, axis=0)
        offsets[1::2] *= -1
        positions = (self._cursor + offsets).astype(np.float32)
        if self._crosshair is None:
            self._crosshair = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineSegmentMaterial(
                    thickness=2.0,
                    color=THEME["primary"],
                    depth_test=False,
                    alpha_mode="blend",
                ),
            )
            self._crosshair.render_order = 100
            self.window.screens[0].scene.main_scene.add(self._crosshair)
        else:
            self._crosshair.geometry.positions.set_data(positions)

    def set_time(self, seconds):
        """Show the volume closest to ``seconds`` in every BOLD run.

        Parameters
        ----------
        seconds : float
            Time from the start of the runs, clipped to the longest run.
        """
        runs = self._functional_visualizations
        duration = max(((v.n_frames - 1) * v.tr for v in runs), default=0.0)
        self._time = float(np.clip(seconds, 0.0, duration))
        for viz in runs:
            viz.set_time(self._time)
        self.request_refresh()

    def _on_run_time(self, viz, seconds):
        """Follow a BOLD run whose own volume slider moved.

        Parameters
        ----------
        viz : Functional4D
            Run that changed volume.
        seconds : float
            Its new time.
        """
        self.set_time(seconds)

    def _step_time(self, reference):
        """Advance one volume of ``reference``, looping at the end of the run.

        Parameters
        ----------
        reference : Functional4D
            Run whose TR sets the step.
        """
        next_time = self._time + reference.tr
        if next_time > (reference.n_frames - 1) * reference.tr + 1e-9:
            next_time = 0.0
        self.set_time(next_time)

    def _start_analysis(self, message, task, on_result, *args):
        """Run an analysis in the background behind the loading overlay.

        Parameters
        ----------
        message : str
            Loader text.
        task : callable
            Function run in a worker thread with ``*args``.
        on_result : callable
            Called on the UI thread with the task's result.
        *args
            Positional arguments for ``task``.
        """
        self._loading_total += 1
        self.loader(True, message=message)

        def _done(result, exception):
            self._loading_done += 1
            if exception is not None:
                logger.error(f"{message.rstrip('.')} failed: {exception}")
                return
            on_result(result)

        run_async(task, _done, *args)

    def _show_stat_map(self, data, affine, name):
        """Replace the data of the overlay called ``name``, or add it.

        Parameters
        ----------
        data : ndarray, shape (X, Y, Z)
            Statistical map.
        affine : ndarray, shape (4, 4)
            Voxel-to-world affine.
        name : str
            Overlay name; an existing overlay with this name is updated.
        """
        for viz in self._stat_map_visualizations:
            if viz.path == name and viz.stat.shape == data.shape:
                viz.set_map(data)
                self.request_refresh()
                return
        self._queue_loaded_visualizations(
            {"stat_maps": [(data, affine, name)], "keep_camera": True},
            message="Adding activation map...",
        )

    def compute_seed_correlation(self, viz):
        """Map the correlation of every voxel of ``viz`` with the cursor's series.

        The result is shown as an activation overlay named
        ``"Seed r · <run>"``, updated in place on later seeds.

        Parameters
        ----------
        viz : Functional4D
            BOLD run to correlate.
        """
        sample = None if self._cursor is None else viz.timeseries_at(self._cursor)
        if sample is None:
            logger.warning("Pick a voxel inside the BOLD run to seed a correlation.")
            return
        voxel, seed = sample
        name = f"Seed r · {Path(viz.name).name.split('.')[0]}"

        def _correlate(bold, seed):
            return seed_correlation(bold, seed, mask=viz.mask)

        self._start_analysis(
            "Computing seed correlation...",
            _correlate,
            lambda r_map: self._show_stat_map(r_map, viz.affine, name),
            viz.bold,
            np.array(seed),
        )

    def compute_glm(self, viz):
        """Fit the run's task events and show one t-map per condition.

        Each map is an activation overlay named ``"t · <condition> · <run>"``,
        updated in place when the GLM is refit.

        Parameters
        ----------
        viz : Functional4D
            BOLD run with task events.
        """
        if not viz.events:
            logger.warning(f"{viz.name} has no task events to fit.")
            return
        stem = Path(viz.name).name.split(".")[0]

        def _fit(bold, events, tr):
            return fit_glm(bold, events, tr, mask=viz.mask)

        def _show(t_maps):
            for condition, t_map in t_maps.items():
                self._show_stat_map(t_map, viz.affine, f"t · {condition} · {stem}")

        self._start_analysis(
            "Fitting task GLM...", _fit, _show, viz.bold, dict(viz.events), viz.tr
        )

    def _on_analysis_request(self, viz, kind):
        """Dispatch an analysis requested from a BOLD run's controls.

        Parameters
        ----------
        viz : Functional4D
            Run the analysis applies to.
        kind : {"glm"}
            Requested analysis.
        """
        if kind == "glm":
            self.compute_glm(viz)

    def _on_cluster_navigate(self, viz, world):
        """Center the slices and cursor on a cluster peak.

        Parameters
        ----------
        viz : StatMap3D
            Overlay whose cluster was chosen.
        world : ndarray, shape (3,)
            Peak coordinate in world millimeters.
        """
        self.set_cursor(world, center=True)

    def _draw_fmri_ui(self):
        """Advance playback; draw the dock, colorbars, view labels and cursor info."""
        now = time.perf_counter()
        for viz in self._stat_map_visualizations:
            viz.tick(now)
        runs = self._functional_visualizations
        dock_height = self._timeseries_panel.height if runs else 0
        if dock_height != self._dock_height:
            self._dock_height = dock_height
            self._apply_screen_layout()
        if runs:
            if self._timeseries_panel.playing and not self._was_playing:
                # Raw volumes barely change; play the percent-change overlay.
                self._show_activity(runs, force=True)
            self._was_playing = self._timeseries_panel.playing
            if self._timeseries_panel.step_due(now):
                self._step_time(runs[0])
            self._timeseries_panel.render(
                (self.ui_size[0], self.size[1] - dock_height),
                (self.size[0] - self.ui_size[0], dock_height),
                layers=runs,
                cursor=self._cursor,
                time_s=self._time,
            )
        self._draw_colorbars()
        self._draw_layout_overlay()
        self._draw_cursor_info()

    def _show_activity(self, runs, *, force=False):
        """Switch runs still in raw ``Volume`` mode to the ``Change`` overlay.

        Runs whose mode the user picked are left alone. Without ``force``,
        runs only switch when an anatomical image is loaded to overlay on.

        Parameters
        ----------
        runs : list of Functional4D
            Candidate runs.
        force : bool, optional
            Switch even without anatomy (used when playback starts).
        """
        has_anatomy = any(
            not isinstance(viz, (Functional4D, StatMap3D))
            for viz in self._image_visualizations
        )
        if not (force or has_anatomy):
            return
        for run in runs:
            if run.display_mode == "Volume" and not run.mode_chosen:
                run.set_display_mode("Change")

    def _draw_colorbars(self):
        """Draw the color scale of up to three visible overlays in the view."""
        maps = [
            viz
            for viz in self._functional_visualizations + self._stat_map_visualizations
            if viz.actor.visible and viz.is_overlay
        ]
        if not maps:
            return
        draw_list = imgui.get_background_draw_list()
        height = min(160.0, (self.size[1] - self._dock_height) * 0.3)
        x = self.size[0] - 30.0
        y = self.size[1] - self._dock_height - height - 20.0
        text_color = imgui.get_color_u32(THEME["secondary"])
        for viz in reversed(maps[-3:]):
            viz.draw_colorbar(
                draw_list, origin=(x, y), size=(12.0, height), vertical=True
            )
            label = viz.name if len(viz.name) <= 18 else f"{viz.name[:17]}…"
            label_size = imgui.calc_text_size(label)
            draw_list.add_text(
                (x + 12 - label_size.x, y - label_size.y - 8), text_color, label
            )
            x -= max(label_size.x, 60.0) + 40.0

    def _screen_bboxes(self):
        """Return the viewport of every active screen, in screen order.

        Returns
        -------
        list of tuple
            ``(x, y, w, h)`` boxes; the 3D view fills the area between the
            sidebar and the dock, or its bottom-right quarter in the
            triplanar layout (axial top-left, sagittal top-right, coronal
            bottom-left).
        """
        x0 = self.ui_size[0] if self.UI_window is not None else 0
        width = self.size[0] - x0
        height = self.size[1] - self._dock_height
        if self._layout != "triplanar":
            return [(x0, 0, width, height)]
        half_w, half_h = width / 2.0, height / 2.0
        corners = {
            "3D": (x0 + half_w, half_h),
            "Axial": (x0, 0),
            "Sagittal": (x0 + half_w, 0),
            "Coronal": (x0, half_h),
        }
        names = ["3D"] + [view["name"] for view in self._planar_views]
        return [(*corners[name], half_w, half_h) for name in names]

    def _apply_screen_layout(self):
        """Fit the viewports between the sidebar and the time-series dock."""
        self.window._screen_config = self._screen_bboxes()
        update_viewports(
            self.window.screens,
            calculate_screen_sizes(
                self.window._screen_config, self.window.renderer.logical_size
            ),
        )

    @property
    def layout(self):
        """Current view layout.

        Returns
        -------
        str
            ``"3d"`` for the single 3D view or ``"triplanar"`` for axial,
            sagittal and coronal 2D views next to the 3D view.
        """
        return self._layout

    def set_layout(self, layout):
        """Switch between the 3D view and the 2D triplanar layout.

        In the triplanar layout each 2D view looks straight at one slice
        plane with an orthographic camera (neurological convention: the
        subject's left on the left), pans with the left button and zooms with
        the wheel. Clicking a 2D view moves every synchronized slice through
        the clicked point.

        Parameters
        ----------
        layout : {"3d", "triplanar"}
            Layout to show.

        Raises
        ------
        ValueError
            If ``layout`` is unknown.
        """
        layout = layout.lower()
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}.")
        if layout == self._layout:
            return
        screens = self.window.screens
        if layout == "triplanar":
            if not self._planar_views:
                self._planar_views = self._create_planar_views()
            for view in self._planar_views:
                view["screen"].controller.enabled = True
                screens.append(view["screen"])
            # Match the 2D cameras' octant so overlays face every panel.
            camera = screens[0].camera
            center = self._scene_center()
            distance = np.linalg.norm(np.asarray(camera.world.position) - center)
            direction = -_PLANAR_SIDES / np.linalg.norm(_PLANAR_SIDES)
            camera.world.reference_up = (0.0, 0.0, 1.0)
            camera.world.position = center - direction * max(distance, 1.0)
            camera.look_at(center)
        else:
            for view in self._planar_views:
                view["screen"].controller.enabled = False
                if view["screen"] in screens:
                    screens.remove(view["screen"])
        self._layout = layout
        self._apply_screen_layout()
        self.active_image and self._arrange_image_actors()
        self.request_refresh()

    def _scene_center(self):
        """Return the center of the loaded data's bounding box (origin if empty)."""
        bbox = self.window.screens[0].scene.main_scene.get_world_bounding_box()
        if bbox is None or not np.isfinite(bbox).all():
            return np.zeros(3)
        return np.asarray(bbox).mean(axis=0)

    def _create_planar_views(self):
        """Build orthographic screens sharing the 3D scene, one per slice axis."""
        from fury.window import Screen

        scene = self.window.screens[0].scene
        center = self._scene_center()
        views = []
        for name, axis, up, letters in _PLANAR_VIEWS:
            camera = gfx.OrthographicCamera()
            camera.world.reference_up = up
            direction = np.zeros(3)
            direction[axis] = _PLANAR_SIDES[axis]
            camera.world.position = center + direction * 500.0
            camera.look_at(center)
            viewport = _SlabViewport(self.window.renderer, self, axis)
            controller = gfx.PanZoomController(camera, register_events=viewport)
            views.append(
                {
                    "name": name,
                    "screen": Screen(viewport, scene, camera, controller),
                    "letters": letters,
                }
            )
        return views

    def _slab_objects(self):
        """Split visible tract, surface and ROI objects for the 2D views.

        Returns
        -------
        clipped : list
            Objects whose (pygfx) material honors clipping planes.
        hidden : list
            Visible objects whose material ignores clipping, such as FURY's
            streamtubes; the 2D views hide them.
        """
        clipped, hidden = [], []
        for viz in (
            self._tractogram_visualizations
            + self._surface_visualizations
            + self._roi_visualizations
        ):
            for obj in viz.actor.iter():
                material = getattr(obj, "material", None)
                if material is None:
                    continue
                if type(material).__module__.startswith("pygfx"):
                    clipped.append(obj)
                elif obj.visible:
                    hidden.append(obj)
        return clipped, hidden

    def _slice_position(self):
        """World position of the synchronized slices (scene center if none)."""
        reference = self._get_global_slice_reference()
        if reference is None:
            return self._scene_center()
        return np.asarray(reference.state[:3], dtype=float)

    def _planar_view_at(self, x, y):
        """Return the 2D view under window point ``(x, y)``, or None."""
        if self._layout != "triplanar":
            return None
        for view in self._planar_views:
            vx, vy, vw, vh = view["screen"].viewport.rect
            if vx <= x < vx + vw and vy <= y < vy + vh:
                return view
        return None

    def _track_pointer(self, event):
        """Remember where the last press happened, to route 2D-view clicks."""
        self._pointer = (event.x, event.y)

    def _toggle_layout(self):
        """Flip between the 3D view and the triplanar layout."""
        self.set_layout("3d" if self._layout == "triplanar" else "triplanar")

    def _draw_layout_overlay(self):
        """Label each triplanar view and mark its anatomical directions."""
        if self._layout != "triplanar":
            return
        draw_list = imgui.get_background_draw_list()
        border = imgui.get_color_u32(THEME["shadow"])
        title = imgui.get_color_u32(THEME["primary"])
        letter = imgui.get_color_u32(THEME["text"])
        for view in [{"name": "3D", "screen": self.window.screens[0]}] + list(
            self._planar_views
        ):
            x, y, w, h = view["screen"].viewport.rect
            draw_list.add_rect((x, y), (x + w, y + h), border, 0.0, 1.0)
            draw_list.add_text((x + 10, y + 8), title, view["name"].upper())
            if "letters" not in view:
                continue
            top, bottom, left, right = view["letters"]
            for text, (px, py) in (
                (top, (x + w / 2, y + 10)),
                (bottom, (x + w / 2, y + h - 22)),
                (left, (x + 8, y + h / 2)),
                (right, (x + w - 18, y + h / 2)),
            ):
                draw_list.add_text((px, py), letter, text)

    def capture_figure(
        self,
        path,
        *,
        scale=2,
        background="scene",
        colorbar=True,
        crosshair=False,
        panels=False,
    ):
        """Render the current view to a high-resolution PNG for figures.

        Every visible view (one in the 3D layout, four in the triplanar
        layout) is re-rendered offscreen at ``scale`` times its on-screen
        size and composed in its on-screen arrangement. The sidebar and the
        time-series panel are not included.

        Parameters
        ----------
        path : str or Path
            Output PNG path (``.png`` is appended when missing).
        scale : int, optional
            Resolution multiplier relative to the window.
        background : {"scene", "white", "black", "transparent"}, optional
            Background color; ``"transparent"`` writes an alpha channel.
        colorbar : bool, optional
            Draw a labelled color scale for each visible overlay.
        crosshair : bool, optional
            Keep the orange cursor cross in the image.
        panels : bool, optional
            In the triplanar layout, also save each view as
            ``<name>_<view>.png``.

        Returns
        -------
        list of str
            Paths written, the composed figure first.

        Raises
        ------
        ValueError
            If ``background`` is unknown or ``scale`` is not positive.
        """
        from PIL import Image

        if background not in FIGURE_BACKGROUNDS:
            raise ValueError(
                f"background must be one of {FIGURE_BACKGROUNDS}, got {background!r}."
            )
        if scale <= 0:
            raise ValueError("scale must be positive.")
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".png")
        # Make sure coplanar layers are stacked for the current camera/layout.
        self.active_image and self._arrange_image_actors()
        scene = self.window.screens[0].scene
        previous_background = self._bg_color
        background_rgb = {
            "scene": tuple(previous_background[:3]),
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "transparent": (0.0, 0.0, 0.0),
        }[background]
        crosshair_visible = self._crosshair is not None and self._crosshair.visible
        if self._crosshair is not None:
            self._crosshair.visible = crosshair
        scene.background = (0, 0, 0, 0)
        try:
            tiles = []
            for name, screen in self._capture_views():
                x, y, w, h = screen.viewport.rect
                size = (max(int(w * scale), 1), max(int(h * scale), 1))
                target = gfx.Texture(dim=2, size=(*size, 1), format="rgba8unorm")
                renderer = gfx.renderers.WgpuRenderer(target, pixel_ratio=1)
                slab = getattr(screen.viewport, "slab", nullcontext)
                with slab():
                    renderer.render(scene.main_scene, screen.camera)
                rgba = np.asarray(renderer.snapshot()).astype(np.float32) / 255.0
                tiles.append((name, (x, y, w, h), rgba))
        finally:
            scene.background = previous_background
            if self._crosshair is not None:
                self._crosshair.visible = crosshair_visible
            self.request_refresh()

        def finish(rgba):
            if background != "transparent":
                alpha = rgba[..., 3:4]
                rgb = rgba[..., :3] * alpha + np.asarray(background_rgb) * (1 - alpha)
                rgba = np.concatenate([rgb, np.ones_like(alpha)], axis=-1)
            return Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8))

        x0 = min(box[0] for _, box, _ in tiles)
        y0 = min(box[1] for _, box, _ in tiles)
        width = max(int((box[0] + box[2] - x0) * scale) for _, box, _ in tiles)
        height = max(int((box[1] + box[3] - y0) * scale) for _, box, _ in tiles)
        figure = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        written = []
        panel_images = []
        for name, box, rgba in tiles:
            image = finish(rgba)
            figure.paste(
                image, (int((box[0] - x0) * scale), int((box[1] - y0) * scale))
            )
            panel_images.append((name, image))
        if colorbar:
            figure = self._draw_figure_colorbars(figure, background_rgb, scale)
        if background != "transparent":
            figure = figure.convert("RGB")
        figure.save(path)
        written.append(str(path))
        if panels and len(panel_images) > 1:
            for name, image in panel_images:
                panel_path = path.with_name(f"{path.stem}_{name.lower()}{path.suffix}")
                image.save(panel_path)
                written.append(str(panel_path))
        logger.info(f"Saved figure to {', '.join(written)}")
        return written

    def _capture_views(self):
        """Return ``(name, screen)`` for every view shown in the current layout."""
        views = [("3D", self.window.screens[0])]
        if self._layout == "triplanar":
            views += [(view["name"], view["screen"]) for view in self._planar_views]
        return views

    def _draw_figure_colorbars(self, figure, background_rgb, scale):
        """Draw a labelled vertical color scale per visible overlay onto ``figure``."""
        from PIL import ImageDraw, ImageFont

        from skyline_fmri.UI.theme import FONT
        from skyline_fmri.render.stat_map import stat_lut

        overlays = [
            viz
            for viz in self._functional_visualizations + self._stat_map_visualizations
            if viz.actor.visible and viz.is_overlay
        ]
        if not overlays:
            return figure
        draw = ImageDraw.Draw(figure)
        font_size = int(13 * scale)
        try:
            font = ImageFont.truetype(str(FONT), font_size)
        except OSError:
            font = ImageFont.load_default()
        dark = sum(background_rgb) < 1.5
        ink = (235, 235, 235, 255) if dark else (30, 30, 30, 255)
        bar_w, bar_h = int(14 * scale), int(min(170 * scale, figure.height * 0.35))
        x = figure.width - int(24 * scale) - bar_w
        y = figure.height - bar_h - int(24 * scale)
        for viz in reversed(overlays[-3:]):
            if isinstance(viz, StatMap3D):
                cmap, thr, vmax, sign = viz.colormap, viz.threshold, viz.vmax, viz.sign
                title = Path(viz.name).name.split(".")[0]
            else:
                cmap, thr, vmax, sign = (
                    "Cold-Hot",
                    viz.change_threshold,
                    viz.change_max,
                    "both",
                )
                title = "% change"
            lut = stat_lut(cmap, thr, vmax, sign, size=bar_h)
            for row in range(bar_h):
                r, g, b, a = lut[bar_h - 1 - row]
                color = (
                    tuple(int(255 * c) for c in (r, g, b)) + (255,)
                    if a > 0
                    else (128, 128, 128, 60)
                )
                draw.line([(x, y + row), (x + bar_w, y + row)], fill=color)
            draw.rectangle(
                [x, y, x + bar_w, y + bar_h], outline=ink, width=max(1, scale // 2)
            )
            marks = [vmax, thr] + ([-thr, -vmax] if sign != "positive" else [])
            if sign == "negative":
                marks = [-thr, -vmax]
            for value in marks:
                py = y + bar_h * (1 - (value + vmax) / (2 * vmax))
                label = f"{value:.2f}"
                text_w = draw.textlength(label, font=font)
                draw.line(
                    [(x - 4 * scale, py), (x, py)], fill=ink, width=max(1, scale // 2)
                )
                draw.text(
                    (x - 7 * scale - text_w, py - font_size / 2),
                    label,
                    font=font,
                    fill=ink,
                )
            title_w = draw.textlength(title, font=font)
            draw.text(
                (x + bar_w - title_w, y - font_size - 8 * scale),
                title,
                font=font,
                fill=ink,
            )
            x -= int(max(title_w, 60 * scale) + 60 * scale)
        return figure

    def _tracts(self):
        """Loaded atlas/tractography layers as ``{name: streamlines}`` (world mm).

        Layers created by isolating a tract are left out.

        Returns
        -------
        dict
            Tract display name to its streamlines.
        """
        return {
            viz.name: viz.sft.streamlines
            for viz in self._tractogram_visualizations
            if viz.name not in self._isolated_tracts
        }

    def _on_tracts_changed(self):
        """Recount tract overlaps and rebuild the cursor tract index."""
        self._tract_index = None
        self._cursor_info_key = None
        for viz in self._stat_map_visualizations:
            viz.refresh_tracts()

    def tracts_at(self, world, *, radius=2.0):
        """Return the loaded tracts passing within ``radius`` mm of ``world``.

        Parameters
        ----------
        world : array-like, shape (3,)
            Point in world millimeters.
        radius : float, optional
            Search radius in millimeters.

        Returns
        -------
        list of str
            Tract display names (file names), in load order.
        """
        if self._tract_index is None:
            self._tract_index = TractIndex(self._tracts())
        return self._tract_index.tracts_at(world, radius=radius)

    def _isolate_tract(self, stat_map, name):
        """Add a layer with the streamlines of ``name`` crossing ``stat_map``'s clusters.

        Parameters
        ----------
        stat_map : StatMap3D
            Overlay whose kept clusters select streamlines.
        name : str
            Display name of the tract layer.
        """
        from dipy.io.stateful_tractogram import StatefulTractogram

        tract = next(
            (viz for viz in self._tractogram_visualizations if viz.name == name), None
        )
        if tract is None:
            return
        affine = np.eye(4) if stat_map.affine is None else stat_map.affine
        keep = np.flatnonzero(
            streamlines_through(tract.sft.streamlines, stat_map.labels, affine)
        )
        if keep.size == 0:
            logger.warning(f"No streamline of {name} crosses the activation.")
            return
        sft = StatefulTractogram.from_sft(tract.sft.streamlines[keep], tract.sft)
        label = f"{tract_key(name)} in {Path(stat_map.name).name.split('.')[0]}"
        self._isolated_tracts.add(label)
        self._queue_loaded_visualizations(
            {"tractograms": [(sft, label)], "keep_camera": True},
            message="Isolating tract...",
        )

    def _cursor_lines(self):
        """Return the cursor readout: position, overlay values and nearby tracts."""
        key = (tuple(np.round(self._cursor, 2)), len(self._tractogram_visualizations))
        if key == self._cursor_info_key:
            return self._cursor_info
        lines = ["x {:.1f}   y {:.1f}   z {:.1f} mm".format(*self._cursor)]
        for viz in self._stat_map_visualizations:
            voxel = voxel_from_world(
                self._cursor,
                np.eye(4) if viz.affine is None else viz.affine,
                viz.stat.shape,
            )
            if voxel is not None:
                lines.append(
                    f"{Path(viz.name).name.split('.')[0]}: {viz.stat[voxel]:.2f}"
                )
        if self._tractogram_visualizations:
            names = [tract_label(name) for name in self.tracts_at(self._cursor)]
            lines.append("Tracts: " + (", ".join(names) if names else "none nearby"))
        self._cursor_info_key, self._cursor_info = key, lines
        return lines

    def _draw_cursor_info(self):
        """Show the cursor position, overlay values and tract names in the view."""
        if self._cursor is None:
            return
        lines = self._cursor_lines()
        draw_list = imgui.get_background_draw_list()
        x, y, _, _ = self.window.screens[0].viewport.rect
        if self._layout == "triplanar":
            x, y = self.ui_size[0], 0
        x, y = x + 12, y + 30
        line_h = imgui.get_text_line_height() + 4
        width = max(imgui.calc_text_size(line).x for line in lines) + 20
        draw_list.add_rect_filled(
            (x, y),
            (x + width, y + line_h * len(lines) + 10),
            imgui.get_color_u32((0.05, 0.05, 0.05, 0.75)),
            6.0,
        )
        for index, line in enumerate(lines):
            color = (
                THEME["primary"] if line.startswith("Tracts") else THEME["secondary"]
            )
            draw_list.add_text(
                (x + 10, y + 6 + index * line_h), imgui.get_color_u32(color), line
            )

    def _scene_op_key(self, func):
        """Build a stable key for deferred scene operation coalescing.

        Parameters
        ----------
        func : callable
            Deferred scene operation callable.

        Returns
        -------
        tuple or None
            Comparable key for the operation, or None when unavailable.
        """
        method = getattr(func, "__func__", None)
        owner = getattr(func, "__self__", None)
        if method is not None and owner is not None:
            return (id(owner), method.__name__)
        name = getattr(func, "__name__", None)
        if name is not None:
            return (None, name)
        return None

    def enqueue_scene_op(self, func, *args, **kwargs):
        """Run or defer a scene-mutating callable.

        Runs ``func`` immediately unless the UI is currently drawing, in
        which case the call is queued for ``_flush_pending_scene_ops`` and
        coalesced with any previously queued call sharing the same bound
        method or function name.

        Parameters
        ----------
        func : callable
            Scene-mutating callable to run or defer.
        *args : tuple
            Positional arguments forwarded to ``func``.
        **kwargs : dict
            Keyword arguments forwarded to ``func``.
        """
        if self._is_drawing_ui:
            op_key = self._scene_op_key(func)
            if op_key is not None:
                for idx in range(len(self._pending_scene_ops) - 1, -1, -1):
                    old_func, _, _ = self._pending_scene_ops[idx]
                    if self._scene_op_key(old_func) == op_key:
                        self._pending_scene_ops[idx] = (func, args, kwargs)
                        self.request_refresh()
                        return
            self._pending_scene_ops.append((func, args, kwargs))
            self.request_refresh()
            return
        func(*args, **kwargs)
        self.request_refresh()

    def _perform_refresh(self):
        """Update the cluster helper, UI bookkeeping, and scene actors.

        Does not render; callers that need the change visible must follow
        up with ``_render_window``.
        """
        if self._visualizer_type != "stealth":
            self._update_tractogram_helper()
            self._refresh_ui()
        self._refresh_actors()

    def _perform_refresh_and_render(self):
        """Perform a refresh, clear the refresh flag, and render the window."""
        self._perform_refresh()
        self._refresh_requested = False
        self._render_window()

    def _render_window(self):
        """Render the window, unless the UI is currently being drawn.

        Rendering while ``draw_ui`` is running is skipped because the
        surrounding ImGui frame already triggers a render.
        """
        if self._is_drawing_ui:
            return
        self.window.render()

    def _queue_loaded_visualizations(self, loaded_files, *, message="Loading Files..."):
        """Queue a batch of already-loaded visualization data for creation.

        The batch is consumed later by ``_drain_pending_visualizations``.

        Parameters
        ----------
        loaded_files : dict
            Mapping with the ``"images"``, ``"peaks"``, ``"rois"``,
            ``"surfaces"``, ``"tractograms"``, and ``"shm_coeffs"`` keys,
            each holding a list of loaded-data tuples.
        message : str, optional
            Message text shown to the user.
        """
        self._pending_loaded_files.append(loaded_files)
        self._loading_total += 1
        self._loading_done += 1
        self.loader(True, message=message)

    def _flush_pending_sync_requests(self):
        """Apply state-synchronization requests queued while drawing the UI.

        Requests are re-checked at flush time in case the source
        visualization had synchronization toggled off while queued.
        """
        if not self._pending_sync_requests:
            return
        pending = self._pending_sync_requests.copy()
        self._pending_sync_requests.clear()
        for source_viz, new_state in pending:
            # Re-check source sync at flush time: user may have toggled it off
            # while the request was queued.
            self._synchronize_visualizations_from_source(source_viz, new_state)
        self.active_image and self._arrange_image_actors()
        self._refresh_requested = True

    def _flush_pending_scene_ops(self):
        """Apply scene operations queued while drawing the UI.

        Each queued callable is invoked with its stored arguments; failures
        are logged rather than propagated so one broken operation does not
        block the others.
        """
        if not self._pending_scene_ops:
            return
        pending = self._pending_scene_ops.copy()
        self._pending_scene_ops.clear()
        for func, args, kwargs in pending:
            try:
                func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Failed to apply deferred scene operation %s: %s",
                    getattr(func, "__qualname__", repr(func)),
                    e,
                )
        self._refresh_requested = True

    def _get_global_slice_reference(self):
        """Return the registered synchronized slice reference."""
        eligible = [
            viz
            for viz in self.visualizations
            if isinstance(viz, (Image3D, Peak3D, SHGlyph3D)) and viz._synchronize
        ]
        if self.active_image in eligible:
            return self.active_image
        if self._slice_focus_viz in eligible:
            return self._slice_focus_viz
        return eligible[-1] if eligible else None

    def _get_global_slice_controls(self):
        """Return the reference, slider values, bounds, and visibility."""
        datasets = (
            self._image_visualizations
            + self._peak_visualizations
            + self._sh_glyph_visualizations
        )
        if not datasets:
            return None
        visibility = [
            any(viz._slice_visibility[axis] for viz in datasets) for axis in range(3)
        ]
        reference = self._get_global_slice_reference()
        if reference is None:
            return None, None, None, visibility
        if isinstance(reference, Image3D):
            shape = reference.dwi.shape[:3]
        elif isinstance(reference, Peak3D):
            shape = reference.peaks.shape[:3]
        else:
            shape = reference.shape
        values = slice_slider_values_from_state(
            reference.state, affine=reference.affine
        )
        bounds = slice_slider_bounds(shape, affine=reference.affine)
        return reference.name, values, bounds, visibility

    def _set_global_slice_position(self, axis, value):
        """Move synchronized slices in the reference's slider coordinates.

        Parameters
        ----------
        axis : int
            X/Y/Z axis index.
        value : float
            Slice slider coordinate.
        """
        reference = self._get_global_slice_reference()
        if reference is None:
            return
        values = slice_slider_values_from_state(
            reference.state, affine=reference.affine
        ).copy()
        values[axis] = round(value)
        new_state = slice_state_from_slider_values(values, affine=reference.affine)
        reference.update_state(new_state)
        self._synchronize_visualizations(reference, new_state)
        self.request_refresh()

    def _set_global_slice_visibility(self, axis, visible):
        """Set one slice axis across all datasets, regardless of synchronization.

        Parameters
        ----------
        axis : int
            X/Y/Z axis index.
        visible : bool
            Whether to show this axis.
        """
        for viz in (
            self._image_visualizations
            + self._peak_visualizations
            + self._sh_glyph_visualizations
        ):
            visibility = viz._slice_visibility.copy()
            visibility[axis] = visible
            if visibility != viz._slice_visibility:
                viz.set_slice_visibility(visibility)
        self.request_refresh()

    def _get_reference_slice_state(self):
        """Return a snapshot of the current slice pose for load-time alignment.

        Returns
        -------
        np.ndarray or list or tuple or None
            Snapshot of slice state from an existing synchronizable visualization,
            or None when no such visualization exists (first load). For an image
            with directions, the current direction is appended as a fourth value.
        """
        if self.active_image is not None:
            return self._reference_state_of(self.active_image)
        if self._slice_focus_viz is not None:
            if self._slice_focus_viz not in self.visualizations:
                self._slice_focus_viz = None
            else:
                return self._reference_state_of(self._slice_focus_viz)
        for viz in reversed(self.visualizations):
            if isinstance(viz, (Image3D, Peak3D, SHGlyph3D)):
                return self._reference_state_of(viz)
        return None

    def _reference_state_of(self, viz):
        """Return the slice state of ``viz``, with its direction if it has one.

        Parameters
        ----------
        viz : Visualization
            Visualization to take the reference state from.

        Returns
        -------
        np.ndarray
            Slice position, followed by the direction index for an image with
            directions.
        """
        state = np.asarray(viz.state, dtype=float)
        if isinstance(viz, Image3D) and viz._has_directions:
            state = np.asarray([*state, viz._volume_idx], dtype=float)
        return self._snapshot_state(state)

    def _apply_reference_slice_state_to_new_visualizations(
        self, reference_state, n_img_before, n_peak_before, n_sh_before
    ):
        """Apply a pre-load slice pose to visualizations created in this batch.

        Parameters
        ----------
        reference_state : array-like or None
            Snapshot from ``_get_reference_slice_state`` before loading, or None.
        n_img_before : int
            Length of ``_image_visualizations`` before ``_load_visualiations``.
        n_peak_before : int
            Length of ``_peak_visualizations`` before ``_load_visualiations``.
        n_sh_before : int
            Length of ``_sh_glyph_visualizations`` before ``_load_visualiations``.
        """
        if reference_state is None:
            return
        new_visualizations = (
            self._image_visualizations[n_img_before:]
            + self._peak_visualizations[n_peak_before:]
            + self._sh_glyph_visualizations[n_sh_before:]
        )
        for viz in new_visualizations:
            viz.update_state(reference_state)

    def _drain_pending_visualizations(self):
        """Consume one queued batch of loaded files into visualizations.

        Pops the oldest pending batch, creates its visualizations while
        preserving the current slice pose, refreshes the scene actors,
        recenters the camera on the updated scene bounds, and hides the
        loader once every queued batch has been consumed.
        """
        if self._pending_loaded_files:
            loaded_files = self._pending_loaded_files.pop(0)
            n_img_before = len(self._image_visualizations)
            n_peak_before = len(self._peak_visualizations)
            n_sh_before = len(self._sh_glyph_visualizations)
            reference_slice_state = self._get_reference_slice_state()
            global_slice_controls = self._get_global_slice_controls()
            self._load_visualiations(
                loaded_files.get("images", []),
                loaded_files.get("peaks", []),
                loaded_files.get("rois", []),
                loaded_files.get("surfaces", []),
                loaded_files.get("tractograms", []),
                loaded_files.get("shm_coeffs", []),
                functional=loaded_files.get("functional", []),
                stat_maps=loaded_files.get("stat_maps", []),
                is_cluster=loaded_files.get("is_cluster_override"),
                async_clustering=loaded_files.get("async_clustering_override"),
            )
            self._apply_reference_slice_state_to_new_visualizations(
                reference_slice_state,
                n_img_before,
                n_peak_before,
                n_sh_before,
            )
            if global_slice_controls is not None:
                visibility = global_slice_controls[3]
                for viz in (
                    self._image_visualizations[n_img_before:]
                    + self._peak_visualizations[n_peak_before:]
                    + self._sh_glyph_visualizations[n_sh_before:]
                ):
                    viz.set_slice_visibility(visibility)

            if self._visualizer_type != "stealth":
                self._update_tractogram_helper()

            self._refresh_actors()
            if (
                not loaded_files.get("keep_camera")
                and self.window.screens[0].scene.main_scene.get_world_bounding_sphere()
                is not None
            ):
                update_camera(
                    self.window.screens[0].camera,
                    None,
                    self.window.screens[0].scene,
                )
                for view in self._planar_views:
                    update_camera(
                        view["screen"].camera, None, self.window.screens[0].scene
                    )
            # The stacking direction depends on where the camera now is.
            self.active_image and self._arrange_image_actors()

        if (
            self._loading_total > 0
            and self._loading_done >= self._loading_total
            and not self._pending_loaded_files
        ):
            self.loader(False)
            self._loading_total = 0
            self._loading_done = 0

    def before_render(self):
        """Refresh and render, or defer to a request if mid-UI-draw.

        Called after the constructor's initial load and whenever a change
        needs to be shown outside of the ``draw_ui`` frame callback.
        """
        if self._is_drawing_ui:
            self.request_refresh()
            return
        self._perform_refresh_and_render()

    def handle_resize(self, size):
        """Update cached layout state after the window is resized.

        Registered as the window's resize callback.

        Parameters
        ----------
        size : tuple of int
            New window size, in pixels, as ``(width, height)``.
        """
        self.size = size
        self.ui_size = (400, self.size[1])
        self.UI_window.size = (self.ui_size[0], size[1])
        self.window._screen_config = [
            (
                self.ui_size[0],
                0,
                self.size[0] - self.ui_size[0],
                self.size[1] - self._dock_height,
            )
        ]
        self.UI_window.size = (self.ui_size[0], size[1])
        self._update_tractogram_helper(remove=True)
        self._render_window()

    def handle_key_events(self, event):
        """Handle playback keys and forward keys to clustered tractograms.

        With BOLD runs loaded, Space plays or pauses and the left and right
        arrows step one volume. Keys typed into an ImGui text field are
        ignored.

        Registered as the renderer's ``"key_down"`` event handler.

        Parameters
        ----------
        event : Event
            Interaction event from the renderer callback.
        """
        runs = self._functional_visualizations
        if runs and self._timeseries_panel is not None:
            if imgui.get_io().want_capture_keyboard:
                return
            if event.key in (" ", "Space"):
                self._timeseries_panel.playing = not self._timeseries_panel.playing
                return
            if event.key in ("ArrowLeft", "ArrowRight"):
                step = runs[0].tr if event.key == "ArrowRight" else -runs[0].tr
                self.set_time(self._time + step)
                return
        for viz in self._tractogram_visualizations:
            if isinstance(viz, ClusterStreamline3D):
                viz.handle_key_events(event)

    def _add_visualization(self, viz):
        """Register a visualization in its per-type list and in the UI.

        Skips registration and logs a warning if a visualization with the
        same path/name id is already present.

        Parameters
        ----------
        viz : Visualization
            Visualization instance to register.

        Raises
        ------
        TypeError
            If ``viz`` is not an instance of a supported visualization type.
        """
        viz_id = f"{viz.path}:{viz.name}"
        if self.UI_window is not None and viz_id in self.UI_window.sections:
            logger.warning(
                f"Visualization with id '{viz_id}' already exists. Skipping."
            )
            return
        if isinstance(viz, Image3D):
            self._image_visualizations.append(viz)
        elif isinstance(viz, Peak3D):
            self._peak_visualizations.append(viz)
        elif isinstance(viz, ROI3D):
            self._roi_visualizations.append(viz)
        elif isinstance(viz, Surface):
            self._surface_visualizations.append(viz)
        elif isinstance(viz, (Streamline3D, ClusterStreamline3D)):
            self._tractogram_visualizations.append(viz)
        elif isinstance(viz, SHGlyph3D):
            self._sh_glyph_visualizations.append(viz)
        else:
            raise TypeError("Unsupported visualization type")
        viz._scene_op_callback = self.enqueue_scene_op
        if isinstance(viz, Image3D):
            viz._pick_callback = self._on_pick
        if self.UI_window is not None:
            self.UI_window.add(viz_id, viz.renderer, viz_type=viz.viz_type)

    def _load_visualiations(
        self,
        images,
        peaks,
        rois,
        surfaces,
        tractograms,
        sh_coeffs,
        *,
        functional=None,
        stat_maps=None,
        is_cluster=None,
        async_clustering=None,
    ):
        """Create and register visualizations from batches of loaded data.

        Each argument is a list of loaded-data tuples in the shape produced
        by ``io.load_files`` (see ``Skyline`` for the tuple forms). Sets the
        last loaded image active and opens the file dialog if no
        visualization ends up loaded.

        Parameters
        ----------
        images : list of tuple, optional
            ``(data, affine)`` or ``(data, affine, filename)`` tuples where
            ``data`` is a nibabel image or ndarray and ``filename`` a display label.
        peaks : list of tuple, optional
            (peak_dirs, affine, filename, peak_values) tuples; see
            `create_peak_visualization`.
        rois : list of tuple, optional
            Loaded ROI data, as ``(roi, affine)`` or
            ``(roi, affine, filename)`` tuples.
        surfaces : list of tuple, optional
            Loaded surface data, as ``(vertices, faces)`` or
            ``(vertices, faces, filename)`` tuples.
        tractograms : list of tuple, optional
            Loaded tractogram data, as ``(sft,)`` or ``(sft, filename)``
            tuples. Entries with no streamlines are skipped with a warning.
        sh_coeffs : list of tuple, optional
            Loaded spherical harmonic coefficient data, as
            ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
            ``(coeffs, affine, filename, basis_type)`` tuples. Entries whose
            ``coeffs`` is not a 4D ndarray with a valid SH count are logged and
            skipped. Explicit ``"standard"`` tuples use a full SH count;
            other inputs use a symmetric count.
        functional : list of tuple, optional
            Loaded BOLD runs, as ``(data, affine)``, ``(data, affine,
            filename)`` or ``(data, affine, filename, tr)`` tuples. The
            viewer's ``tr`` and ``events`` apply to each.
        stat_maps : list of tuple, optional
            Loaded statistical maps, as ``(data, affine)`` or
            ``(data, affine, filename)`` tuples, added after every other layer
            so they overlay it.
        is_cluster : bool, optional
            Overrides ``self._is_cluster`` for the tractograms in this batch.
        async_clustering : bool, optional
            Overrides the default async-clustering choice for the
            tractograms in this batch.
        """
        for idx, input in enumerate(images or []):
            image3d = create_image_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                sync_callabck=self._synchronize_visualizations,
                rgb=self._rgb,
            )
            self._add_visualization(image3d)
        for idx, input in enumerate(functional or []):
            run = create_functional_visualization(
                input,
                idx,
                tr=self._tr,
                events=self._events,
                render_callback=self.request_refresh,
                sync_callabck=self._synchronize_visualizations,
                time_callback=self._on_run_time,
                analysis_callback=self._on_analysis_request,
            )
            run.set_time(self._time)
            self._add_visualization(run)
        for idx, input in enumerate(peaks or []):
            peak3d = create_peak_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                sync_callabck=self._synchronize_visualizations,
            )
            self._add_visualization(peak3d)
        for idx, input in enumerate(rois or []):
            color = next(self._color_gen)
            roi3d = create_roi_visualization(
                input,
                idx,
                color=color,
                render_callback=self.request_refresh,
            )
            self._add_visualization(roi3d)
        for idx, input in enumerate(surfaces or []):
            color = next(self._color_gen) if not self._glass_brain else (0, 0, 0)
            opacity = 25 if self._glass_brain else 100
            surface3d = create_surface_visualization(
                input,
                idx,
                color=color,
                material="basic" if self._glass_brain else "phong",
                opacity=opacity,
                render_callback=self.request_refresh,
            )
            self._add_visualization(surface3d)
        for idx, input in enumerate(tractograms or []):
            if isinstance(input, tuple):
                sft = input[0]
                filename = f"Streamlines {idx}"
                if len(input) == 2:
                    filename = input[1]
                if len(sft.streamlines) <= 0:
                    logger.warning(
                        f"The provide file: {filename} does not "
                        "contain any streamlines."
                    )
                    continue
            tractogram3d = create_streamline_visualization(
                input,
                idx,
                is_cluster=is_cluster if is_cluster is not None else self._is_cluster,
                thr=self._cluster_thr,
                line_type="Line" if self._is_light_version else "Tube",
                render_callback=self.request_refresh,
                colormap=self._color_gen,
                tract_colors=self._tract_colors,
                switch_render_callback=self._update_tractogram_rendering,
                loader=self.loader,
                size_threshold=self._cluster_size_thr,
                length_threshold=self._cluster_length_thr,
                buan_pvals_file=self._buan_pvals,
                async_clustering=(
                    async_clustering
                    if async_clustering is not None
                    else self._direct_load and self._visualizer_type != "stealth"
                ),
            )
            self._add_visualization(tractogram3d)
        for idx, input in enumerate(sh_coeffs or []):
            if isinstance(input, tuple):
                coeffs = input[0]
                filename = f"ODFs {idx}"
                if len(input) >= 3:
                    filename = input[2]
                full_basis = len(input) == 4 and input[3] == "standard"
                if not _valid_shm_coeffs(coeffs, filename, full_basis=full_basis):
                    continue
            sh3d = create_shm_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                scale=1.0,
                l_max=8,
                basis_type=self._sh_basis,
                sync_callback=self._synchronize_visualizations,
            )
            self._add_visualization(sh3d)

        for idx, input in enumerate(stat_maps or []):
            stat_map = create_stat_map_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                sync_callabck=self._synchronize_visualizations,
                navigate_callback=self._on_cluster_navigate,
                tract_provider=self._tracts,
                isolate_callback=self._isolate_tract,
            )
            self._add_visualization(stat_map)

        self._show_activity(self._functional_visualizations)
        if tractograms:
            self._on_tracts_changed()

        if self._image_visualizations:
            self._image_visualizations[-1].active = True
            self.active_image = self._image_visualizations[-1]
            self._arrange_image_actors()

        if len(self.visualizations) == 0 and self.UI_window is not None:
            self.UI_window.request_file_dialog = True

    def _append_visualization(
        self,
        *,
        filenames=None,
        rois=None,
        peaks=None,
        shm_coeffs=None,
        functional=None,
        stat_maps=None,
        events=None,
        sh_basis=None,
    ):
        """Load files from disk asynchronously and queue them for display.

        Each path is loaded in its own background task via
        ``io.load_files``; each completed task's result is appended to
        ``self._pending_loaded_files`` for ``_drain_pending_visualizations``
        to consume on a later frame.

        Parameters
        ----------
        filenames : list of str, optional
            Paths to images, peaks, surfaces, or tractograms to load.
        rois : list of str, optional
            Paths to ROI files to load.
        peaks : list, optional
            List of ``.pam5`` or NIfTI (.nii, .nii.gz) peak file paths.
        shm_coeffs : list of str, optional
            Paths to spherical harmonic coefficient files to load.
        functional : list of str, optional
            Paths to 4D BOLD runs to load.
        stat_maps : list of str, optional
            Paths to statistical maps to load as activation overlays.
        events : list of str, optional
            BIDS events tables; the first one is read now and attached to
            every loaded BOLD run and to runs loaded later.
        sh_basis : str or None, optional
            Source convention for this batch, or the viewer's input default.
            Unsupported values are logged and use latest Descoteaux07.
        """
        if events:
            resolved = self._resolve_events(events)
            if resolved is not None:
                self._events = resolved
                source = events[0] if isinstance(events, (list, tuple)) else events
                for run in self._functional_visualizations:
                    run.set_events(resolved, source=str(source))
        batch_sh_basis = self._sh_basis if sh_basis is None else sh_basis
        batch_sh_basis = _validate_sh_basis(batch_sh_basis)
        batches = (
            [{"filenames": [path]} for path in filenames or []]
            + [{"rois": [path]} for path in rois or []]
            + [{"peaks": [path]} for path in peaks or []]
            + [{"shm_coeffs": [path]} for path in shm_coeffs or []]
            + [{"functional": [path]} for path in functional or []]
            + [{"stat_maps": [path]} for path in stat_maps or []]
        )
        if not batches:
            return

        self._loading_total = len(batches)
        self._loading_done = 0

        def load_files_task(*, filenames=None, **paths):
            return load_files(filenames or [], sh_basis=batch_sh_basis, **paths)

        def on_files_loaded(loaded_files, exception):
            self._loading_done += 1
            if exception is not None:
                logger.error(f"Could not load files: {exception}")
            elif loaded_files is not None:
                self._pending_loaded_files.append(loaded_files)

        self.loader(True, message="Loading Files...")
        for batch in batches:
            run_async(load_files_task, on_files_loaded, **batch)

    def _remove_visualization(self, viz):
        """Unregister a visualization from its per-type tracking list.

        Also clears ``self._slice_focus_viz`` if it pointed at ``viz`` and
        opens the file dialog if no visualization remains.

        Parameters
        ----------
        viz : Visualization
            Visualization instance to unregister.

        Raises
        ------
        TypeError
            If ``viz`` is not an instance of a supported visualization type.
        """
        if isinstance(viz, Image3D):
            self._image_visualizations.remove(viz)
            if viz is self.active_image:
                self.active_image = (
                    self._image_visualizations[-1]
                    if self._image_visualizations
                    else None
                )
                if self.active_image is not None:
                    self.active_image.active = True
        elif isinstance(viz, Peak3D):
            self._peak_visualizations.remove(viz)
        elif isinstance(viz, ROI3D):
            self._roi_visualizations.remove(viz)
        elif isinstance(viz, Surface):
            self._surface_visualizations.remove(viz)
        elif isinstance(viz, (Streamline3D, ClusterStreamline3D)):
            self._isolated_tracts.discard(viz.name)
            self._tractogram_visualizations.remove(viz)
            self._on_tracts_changed()
        elif isinstance(viz, SHGlyph3D):
            self._sh_glyph_visualizations.remove(viz)
        else:
            raise TypeError("Unsupported visualization type")

        if viz is self._slice_focus_viz:
            self._slice_focus_viz = None
        if viz is self.active_image:
            self.active_image = (
                self._image_visualizations[-1] if self._image_visualizations else None
            )
            for image in self._image_visualizations:
                image.active = image is self.active_image

        if len(self.visualizations) == 0 and self.UI_window is not None:
            self.UI_window.request_file_dialog = True

    @staticmethod
    def _snapshot_state(new_state):
        """Copy a slice state so later mutation cannot affect the snapshot.

        Parameters
        ----------
        new_state : array-like
            New synchronized state for this visualization.

        Returns
        -------
        np.ndarray
            The snapshot state of the visualization.
        """
        if hasattr(new_state, "copy"):
            return new_state.copy()
        if isinstance(new_state, list):
            return list(new_state)
        if isinstance(new_state, tuple):
            return tuple(new_state)
        return new_state

    def _synchronize_visualizations_from_source(self, source_viz, new_state):
        """Push a new slice state from ``source_viz`` to other visualizations.

        Parameters
        ----------
        source_viz : Visualization
            Visualization whose state change is being propagated.
        new_state : array-like
            New synchronized state for this visualization.
        """
        # Source-side guard: only push if this view has sync enabled.
        if not getattr(source_viz, "_synchronize", True):
            return

        for viz in self.visualizations:
            if viz is not source_viz and isinstance(viz, (Image3D, Peak3D, SHGlyph3D)):
                # Target-side guard is inside each viz's update_state.
                viz.update_state(new_state)

    def _synchronize_visualizations(self, source_viz, new_state):
        """Propagate a slice-state change reported by a visualization.

        Called by visualizations as their state-change callback. Updates
        the slice-focus visualization, then either queues the request for
        ``_flush_pending_sync_requests`` when the UI is mid-draw, or
        propagates it immediately otherwise.

        Parameters
        ----------
        source_viz : Visualization
            Visualization whose state change is being reported.
        new_state : array-like
            New synchronized state for this visualization.
        """
        if not getattr(source_viz, "_synchronize", True):
            return

        if isinstance(source_viz, (Image3D, Peak3D, SHGlyph3D)):
            self._slice_focus_viz = source_viz

        new_state = self._snapshot_state(new_state)

        if self._is_drawing_ui:
            self._pending_sync_requests.append((source_viz, new_state))
            self.request_refresh()
            return
        self._synchronize_visualizations_from_source(source_viz, new_state)
        self.active_image and self._arrange_image_actors()

    def _update_background_color(self, new_color):
        """Apply or defer a background color change from the UI.

        Parameters
        ----------
        new_color : tuple of float
            New scene background color as an RGB tuple in ``[0, 1]``.
        """
        if self._is_drawing_ui:
            self._pending_bg_color = new_color
            self.request_refresh()
            return
        self._bg_color = new_color
        self.window.screens[0].scene.background = self._bg_color
        self._render_window()

    def _process_tractogram_switches(self):
        """Apply queued clustered/unclustered tractogram mode switches.

        For each queued switch, removes the old visualization from the
        scene and UI, then asynchronously re-creates it in the requested
        mode via a deferred ``run_async`` call.
        """
        if not self._pending_tractogram_switches:
            return
        pending = self._pending_tractogram_switches.copy()
        self._pending_tractogram_switches.clear()
        for viz, is_clustered in pending:
            if viz not in self._tractogram_visualizations:
                continue
            viz_id = f"{viz.path}:{viz.name}"
            sft = viz.sft
            path = viz.path

            if self.UI_window is not None:
                self.UI_window.remove(viz_id)
            # Remove the old visualization object immediately so mode switches
            # replace state instead of coexisting under the same UI id.
            self._remove_visualization(viz)
            self.request_refresh()

            self._loading_total = 1
            self._loading_done = 0

            def _delay():
                pass

            def _on_delay_done(
                _, exception, *, _sft=sft, _path=path, _is_clustered=is_clustered
            ):
                self._loading_done += 1
                self._pending_loaded_files.append(
                    {
                        "images": [],
                        "peaks": [],
                        "rois": [],
                        "surfaces": [],
                        "tractograms": [(_sft, _path)],
                        "shm_coeffs": [],
                        "is_cluster_override": _is_clustered,
                        "async_clustering_override": True,
                    }
                )

            run_async(_delay, _on_delay_done)

    def _update_tractogram_rendering(self, streamline_viz, is_clustered):
        """Queue a clustered/unclustered mode switch for a tractogram.

        Parameters
        ----------
        streamline_viz : Visualization
            Streamline visualization whose rendering mode changed.
        is_clustered : bool
            Whether the visualization should switch to clustered mode.
        """
        for viz in self._tractogram_visualizations:
            if viz is streamline_viz and isinstance(
                viz, (Streamline3D, ClusterStreamline3D)
            ):
                self._pending_tractogram_switches.append((viz, is_clustered))
                break

    def loader(self, show, *, message=None):
        """Show or hide the UI's loading indicator.

        Parameters
        ----------
        show : bool
            Whether to show the UI element/loader.
        message : str, optional
            Message text shown to the user.
        """
        if self.UI_window is not None:
            self.UI_window.update_loader(show=show, message=message)

    def _save_snapshot(self, snapshot_path):
        """Save a figure of the current views (see :meth:`capture_figure`).

        The options chosen in the sidebar's capture popup are used when the
        viewer has a UI; defaults otherwise.

        Parameters
        ----------
        snapshot_path : str
            Target PNG path; ``.png`` is appended when missing.
        """
        _, extension = split_filename_extension(snapshot_path)
        if extension == "":
            snapshot_path = f"{snapshot_path}.png"
        options = dict(self.UI_window.capture_options) if self.UI_window else {}
        logger.info(f"Saving figure to {snapshot_path}")
        self.enqueue_scene_op(self.capture_figure, snapshot_path, **options)

    def _layout_ui(self, *, toggle=False):
        """Report the layout to the sidebar button and toggle it on request.

        Parameters
        ----------
        toggle : bool, optional
            Switch layouts after the current frame.

        Returns
        -------
        str
            The current layout.
        """
        if toggle:
            self.enqueue_scene_op(self._toggle_layout)
        return self._layout

    @property
    def visualizations(self):
        """Return every visualization currently tracked by the viewer.

        Returns
        -------
        list
            The list of visualizations in the Skyline viewer.
        """
        return (
            self._image_visualizations
            + self._peak_visualizations
            + self._roi_visualizations
            + self._surface_visualizations
            + self._tractogram_visualizations
            + self._sh_glyph_visualizations
        )


def skyline_from_files(
    fnames,
    *,
    rois=None,
    peaks=None,
    shm_coeffs=None,
    functional=None,
    stat_maps=None,
    events=None,
    tr=None,
    sh_basis="descoteaux07",
    is_cluster=False,
    is_light_version=False,
    glass_brain=False,
    bg_color=None,
    tract_colors=None,
    cluster_thr=15.0,
    cluster_size_thr=None,
    cluster_length_thr=None,
    buan_pvals=None,
    stealth=False,
    rgb=None,
    layout="3d",
    out_dir=None,
    out_stealth_png=None,
):
    """Launch the Skyline GUI from file paths.

    Loads every path in the background and constructs the corresponding
    ``Skyline`` viewer, forwarding ``fnames``/``rois``/``shm_coeffs`` as
    ``initial_filenames``/``initial_rois``/``initial_shm_coeffs``.

    Parameters
    ----------
    fnames : list of str
        File paths to be loaded into the Skyline viewer.

        Supported file types include:

        - NIfTI images (.nii, .nii.gz)
        - Peaks (.pam5)
        - Surfaces (.pial, .gii, .gii.gz)
        - Tractograms (.trk, .trx, .dpy, .tck, .vtk, .vtp, .fib)

        Unsupported extensions are logged and skipped; ``.npy`` entries are
        ignored.
    rois : list of str, optional
        File paths for ROIs to be loaded into the Skyline viewer. Only
        NIfTI images (.nii, .nii.gz) are supported; other extensions are
        logged and skipped.
    peaks : list, optional
        Tuple of path for each peaks file (.pam5, or NIfTI with shape
        (X, Y, Z, 3*N) or (X, Y, Z, N, 3)) to be added to the Skyline viewer.
    shm_coeffs : list of str, optional
        File paths for spherical harmonics coefficients (``.pam5``, ``.nii``,
        or ``.nii.gz``). Unsupported extensions are logged and skipped.
    functional : list of str, optional
        4D BOLD file paths (``.nii``, ``.nii.gz``). The repetition time comes
        from a BIDS sidecar, the NIfTI header, or ``tr``.
    stat_maps : list of str, optional
        3D statistical map paths (``.nii``, ``.nii.gz``) shown as thresholded
        activation overlays.
    events : str, optional
        BIDS events table (``.tsv``/``.csv``) attached to every BOLD run.
    tr : float, optional
        Repetition time in seconds for every BOLD run, overriding the BIDS
        sidecar and NIfTI header.
    sh_basis : str, optional
        Input convention for PAM and NIfTI ODFs: ``"descoteaux07"`` (latest),
        ``"descoteaux07_legacy"`` (old DIPY), ``"tournier07"`` (MRtrix 0.2),
        or ``"tournier19"`` (MRtrix3).
        Unsupported values are logged and use latest Descoteaux07.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
        which improves performance for large tractograms.
    glass_brain : bool, optional
        Whether to render surfaces black with the ``"basic"`` material at
        25% opacity and default the background to white.
    bg_color : tuple of float, optional
        Background color of the scene as an RGB tuple in ``[0, 1]``. If
        None, it is white when ``glass_brain`` is True, otherwise dark
        gray.
    tract_colors : str or tuple of float or None, optional
        Coloring scheme for the tractograms: ``"direction"`` for
        directionally colored streamlines, ``"random"`` for the next color
        from a distinguishable colormap per tractogram, an RGB(A) tuple in
        ``[0, 1]``, or a string of three space-separated numbers parsed to
        such a tuple. If None, ``"direction"`` is used.
    cluster_thr : float, optional
        Final distance threshold, in mm, used by ``qbx_and_merge`` when
        clustering is enabled; small-animal data may need a smaller value
        such as 2.0.
    cluster_size_thr : int, optional
        Clusters with size less than ``cluster_size_thr`` are hidden. If
        None, the 50th percentile of the cluster size distribution is
        used.
    cluster_length_thr : float, optional
        Clusters with average length less than ``cluster_length_thr`` mm
        are hidden. If None, the 25th percentile of the cluster length
        distribution is used.
    buan_pvals : str, optional
        File path for BUAN p-values used for BUAN-based coloring of
        tractograms.
    stealth : bool, optional
        Request offscreen capture. ``DIPY_OFFSCREEN=1`` or ``true``
        (case-insensitive) forces capture; otherwise, ``stealth`` controls rendering.
    rgb : bool or None, optional
        ``None``: auto-detect from structured NIfTI ``DT_RGB24``
        dtype; show toggle for other 4D volumes with 3 or 4 channels.
        ``True``: force RGB mode.  ``False``: never treat as RGB.
    layout : {"3d", "triplanar"}, optional
        Initial view layout: the 3D view, or 2D axial, sagittal and coronal
        views next to the 3D view.
    out_dir : str or Path, optional
        Directory for the offscreen output image; created if missing.
    out_stealth_png : str, optional
        Output image name used as the offscreen window title. Applies when
        ``stealth`` is True or ``DIPY_OFFSCREEN`` is ``1``/``true``.

    Returns
    -------
    Skyline
        The constructed viewer, returned once construction returns from
        its blocking ``self.window.start()`` call.

    Notes
    -----
    Window creation sets ``FURY_OFFSCREEN`` to ``1`` for stealth or removes
    the key otherwise. Skyline restores its previous value or absence after
    setup/start returns or raises.
    """
    visualizer_type = "stealth" if stealth else "standalone"

    return skyline(
        visualizer_type=visualizer_type,
        initial_filenames=fnames,
        initial_rois=rois,
        initial_peaks=peaks,
        initial_shm_coeffs=shm_coeffs,
        initial_functional=functional,
        initial_stat_maps=stat_maps,
        events=events,
        tr=tr,
        sh_basis=sh_basis,
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
        tract_colors=tract_colors,
        cluster_thr=cluster_thr,
        cluster_size_thr=cluster_size_thr,
        cluster_length_thr=cluster_length_thr,
        buan_pvals=buan_pvals,
        rgb=rgb,
        layout=layout,
        out_dir=out_dir,
        out_stealth_png=out_stealth_png,
    )


def skyline(
    *,
    visualizer_type="standalone",
    images=None,
    peaks=None,
    rois=None,
    surfaces=None,
    tractograms=None,
    sh_coeffs=None,
    functional=None,
    stat_maps=None,
    events=None,
    tr=None,
    sh_basis="descoteaux07",
    is_cluster=False,
    is_light_version=False,
    glass_brain=False,
    bg_color=None,
    tract_colors=None,
    cluster_thr=15.0,
    cluster_size_thr=None,
    cluster_length_thr=None,
    buan_pvals=None,
    rgb=None,
    initial_filenames=None,
    initial_rois=None,
    initial_peaks=None,
    initial_shm_coeffs=None,
    initial_functional=None,
    initial_stat_maps=None,
    layout="3d",
    out_dir=None,
    out_stealth_png=None,
):
    """Launch the Skyline GUI.

    Constructs and returns a ``Skyline`` viewer with the given data.

    Parameters
    ----------
    visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
        Kind of window to create. The options map to FURY window types via
        ``create_window``:

        - "standalone": a default interactive window.
        - "gui": a Qt-based window.
        - "jupyter": an inline Jupyter notebook window.
        - "stealth": an offscreen window with no GUI, used for scripted
          snapshots.

        An unrecognized value logs an error and terminates the process.
        ``DIPY_OFFSCREEN=1`` or ``true`` (case-insensitive) forces stealth.
        Otherwise, ``visualizer_type`` controls rendering.
    images : list of tuple, optional
        ``(data, affine)`` or ``(data, affine, filename)`` tuples where
        ``data`` is a nibabel image or ndarray and ``filename`` is a display label.
    peaks : list of tuple, optional
        (peak_dirs, affine, filename, peak_values) tuples; see
        `create_peak_visualization`.
    rois : list of tuple, optional
        Already-loaded ROI data to show at startup, as ``(roi, affine)`` or
        ``(roi, affine, filename)`` tuples.
    surfaces : list of tuple, optional
        Already-loaded surface data to show at startup, as
        ``(vertices, faces)`` or ``(vertices, faces, filename)`` tuples.
    tractograms : list of tuple, optional
        Already-loaded tractogram data to show at startup, as ``(sft,)`` or
        ``(sft, filename)`` tuples, where ``sft`` is a
        ``StatefulTractogram``. Entries with no streamlines are skipped
        with a warning.
    sh_coeffs : list of tuple, optional
        Already-loaded spherical harmonic coefficient data to show at
        startup, as ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
        ``(coeffs, affine, filename, basis_type)`` tuples. ``coeffs`` must
        be a 4D ndarray with a symmetric SH count, or a full SH count when the
        fourth tuple element is ``"standard"``. Invalid entries are logged
        and skipped.
    functional : list of tuple, optional
        Already-loaded 4D BOLD runs as ``(data, affine)``,
        ``(data, affine, filename)`` or ``(data, affine, filename, tr)``
        tuples, where ``data`` is a nibabel image or ndarray and ``tr`` the
        repetition time in seconds (or None).
    stat_maps : list of tuple, optional
        Already-loaded 3D statistical maps (t, z, correlation, ...) shown as
        thresholded activation overlays, as ``(data, affine)`` or
        ``(data, affine, filename)`` tuples.
    events : str, list of str or dict, optional
        BIDS events table (``.tsv``/``.csv``; the first path of a list), or a
        mapping of condition name to ``(N, 2)`` ``(onset, duration)`` arrays
        in seconds, attached to every BOLD run.
    tr : float, optional
        Repetition time in seconds for every BOLD run, overriding the BIDS
        sidecar and NIfTI header.
    sh_basis : str, optional
        Input convention for ODF files and unlabelled arrays: ``"descoteaux07"``
        (latest), ``"descoteaux07_legacy"`` (old DIPY), ``"tournier07"``
        (MRtrix 0.2), or ``"tournier19"`` (MRtrix3). A fourth tuple element
        overrides this selection.
        Unsupported values are logged and use latest Descoteaux07.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
        which improves performance for large tractograms.
    glass_brain : bool, optional
        Whether to render surfaces black with the ``"basic"`` material at
        25% opacity and default the background to white.
    bg_color : tuple of float, optional
        Background color of the scene as an RGB tuple in ``[0, 1]``. If
        None, it is white when ``glass_brain`` is True, otherwise dark
        gray.
    tract_colors : str or tuple of float or None, optional
        Coloring scheme for the tractograms: ``"direction"`` for
        directionally colored streamlines, ``"random"`` for the next color
        from a distinguishable colormap per tractogram, an RGB(A) tuple in
        ``[0, 1]``, or a string of three space-separated numbers parsed to
        such a tuple. If None, ``"direction"`` is used.
    cluster_thr : float, optional
        Final distance threshold, in mm, used by ``qbx_and_merge`` when
        clustering is enabled; small-animal data may need a smaller value
        such as 2.0.
    cluster_size_thr : int, optional
        Clusters with size less than ``cluster_size_thr`` are hidden. If
        None, the 50th percentile of the cluster size distribution is
        used.
    cluster_length_thr : float, optional
        Clusters with average length less than ``cluster_length_thr`` mm
        are hidden. If None, the 25th percentile of the cluster length
        distribution is used.
    buan_pvals : str, optional
        File path for BUAN p-values used for BUAN-based coloring of
        tractograms.
    rgb : bool or None, optional
        ``None``: auto-detect from structured NIfTI ``DT_RGB24``
        dtype; show toggle for other 4D volumes with 3 or 4 channels.
        ``True``: force RGB mode.  ``False``: never treat as RGB.
    initial_filenames : list of str, optional
        File paths loaded asynchronously into the viewer on startup. If
        neither preloaded data nor initial files are given and a UI
        exists, the file dialog opens on start.
    initial_rois : list of str, optional
        ROI file paths loaded asynchronously into the viewer on startup.
    initial_peaks : list, optional
        List of ``.pam5`` or NIfTI (.nii, .nii.gz) peak file paths to load
        into the Skyline viewer on startup.
    initial_shm_coeffs : list of str, optional
        Spherical harmonic coefficient file paths loaded asynchronously
        into the viewer on startup.
    initial_functional : list of str, optional
        4D BOLD file paths loaded asynchronously into the viewer on startup.
    initial_stat_maps : list of str, optional
        Statistical map file paths loaded asynchronously into the viewer on
        startup.
    layout : {"3d", "triplanar"}, optional
        Initial view layout: the 3D view, or 2D axial, sagittal and coronal
        views next to the 3D view.
    out_dir : str or Path, optional
        Directory for the offscreen output image; created if missing.
    out_stealth_png : str, optional
        Output image name used as the offscreen window title. Applies when
        stealth mode is requested or ``DIPY_OFFSCREEN`` is ``1``/``true``.

    Returns
    -------
    Skyline
        The constructed viewer, returned once construction returns from
        its blocking ``self.window.start()`` call.

    Notes
    -----
    Window creation sets ``FURY_OFFSCREEN`` to ``1`` for stealth or removes
    the key otherwise. Skyline restores its previous value or absence after
    setup/start returns or raises.

    Global visibility affects all slice datasets; positions affect synchronized
    datasets only. Individual visibility changes remain local.
    Titles select layers; trailing arrows toggle expansion.
    """
    return Skyline(
        visualizer_type=visualizer_type,
        images=images,
        peaks=peaks,
        rois=rois,
        surfaces=surfaces,
        tractograms=tractograms,
        sh_coeffs=sh_coeffs,
        functional=functional,
        stat_maps=stat_maps,
        events=events,
        tr=tr,
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
        tract_colors=tract_colors,
        cluster_thr=cluster_thr,
        cluster_size_thr=cluster_size_thr,
        cluster_length_thr=cluster_length_thr,
        buan_pvals=buan_pvals,
        rgb=rgb,
        initial_filenames=initial_filenames,
        initial_rois=initial_rois,
        initial_peaks=initial_peaks,
        initial_shm_coeffs=initial_shm_coeffs,
        initial_functional=initial_functional,
        initial_stat_maps=initial_stat_maps,
        sh_basis=sh_basis,
        layout=layout,
        out_dir=out_dir,
        out_stealth_png=out_stealth_png,
    )


if not has_fury_v2:
    Skyline = skyline = skyline_from_files = fury
