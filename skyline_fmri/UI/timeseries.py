"""Docked time-series panel for Skyline's BOLD runs."""

import time

import numpy as np

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from skyline_fmri.UI.elements import (
    render_file_dialog,
    segmented_switch,
    thin_slider,
    toggle_button,
)
from skyline_fmri.UI.theme import CONDITION_COLORS, THEME
from skyline_fmri.fmri import (
    fitted_response,
    normalize_timeseries,
    power_spectrum,
)

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui
    implot = imgui_bundle.implot
    icons_fontawesome_6 = imgui_bundle.icons_fontawesome_6

#: Line colors cycled through BOLD runs.
LINE_COLORS = (
    (0.92, 0.92, 0.92, 1.0),
    (0.31, 0.76, 0.97, 1.0),
    (0.51, 0.78, 0.52, 1.0),
    (0.73, 0.41, 0.78, 1.0),
)

# segmented_switch title-cases its options, so labels must already be title case.
_NORMALIZATION_LABELS = {"Raw": "raw", "% Signal": "percent", "Z-Score": "zscore"}
_Y_LABELS = {"raw": "signal", "percent": "% change", "zscore": "z"}

HEADER_HEIGHT = 34
EXPANDED_HEIGHT = 290


def _legend_label(name, *, max_chars=28):
    """Return a run name without extensions, shortened for plot legends."""
    stem = name.split("/")[-1].split(".")[0]
    return stem if len(stem) <= max_chars else f"{stem[: max_chars - 1]}…"


class TimeseriesPanel:
    """Plot the BOLD signal under the cursor, with task events and playback.

    The panel is drawn as a borderless ImGui window docked under the 3D
    view. It shows one line per visible BOLD run, shades task events, can
    overlay the task-model fit, switches to a power spectrum, scrubs time
    with a draggable cursor, and plays runs as a movie.

    Parameters
    ----------
    time_callback : callable
        Called as ``time_callback(seconds)`` when the user moves in time.
    seed_callback : callable
        Called as ``seed_callback(viz)`` to correlate ``viz`` with the series
        under the cursor.
    """

    def __init__(self, *, time_callback, seed_callback):
        """Initialize the panel state.

        Parameters
        ----------
        time_callback : callable
            Called with the new time in seconds.
        seed_callback : callable
            Called with the BOLD layer to seed a correlation map from.
        """
        self._time_callback = time_callback
        self._seed_callback = seed_callback
        self.normalization = "percent"
        self.detrend = True
        self.show_fit = True
        self.spectrum = False
        self.collapsed = False
        self.playing = False
        self.fps = 8
        self._last_step = 0.0
        self._fit_view = True
        self._last_cursor = None
        self._implot_context = None
        self._cache = {}

    @property
    def height(self):
        """Current docked height in pixels.

        Returns
        -------
        int
            Header-only height when collapsed, full height otherwise.
        """
        return HEADER_HEIGHT if self.collapsed else EXPANDED_HEIGHT

    def step_due(self, now):
        """Return whether playback should advance one volume.

        Parameters
        ----------
        now : float
            Current ``time.perf_counter()`` value.

        Returns
        -------
        bool
            True at most ``fps`` times per second while playing.
        """
        if not self.playing or now - self._last_step < 1.0 / self.fps:
            return False
        self._last_step = now
        return True

    def series(self, layer, world):
        """Return the displayed series of ``layer`` at ``world``.

        Parameters
        ----------
        layer : Functional4D
            BOLD run.
        world : array-like, shape (3,)
            Cursor in world millimeters.

        Returns
        -------
        dict or None
            ``voxel``, ``raw``, ``values`` (normalized) and, when the run has
            events and fits are shown, ``fit``; None outside the run.
        """
        key = (
            id(layer),
            tuple(np.round(world, 3)),
            self.normalization,
            self.detrend,
            self.show_fit,
            layer.tr,
            id(layer.events),
        )
        if key in self._cache:
            return self._cache[key]
        sample = layer.timeseries_at(world)
        if sample is None:
            result = None
        else:
            voxel, raw = sample
            values = normalize_timeseries(
                raw, mode=self.normalization, detrend=self.detrend
            )
            result = {"voxel": voxel, "raw": np.asarray(raw), "values": values}
            if self.show_fit and layer.events:
                result["fit"] = fitted_response(values, layer.events, layer.tr)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = result
        return result

    def render(self, pos, size, *, layers, cursor, time_s):
        """Draw the panel for the current frame.

        Parameters
        ----------
        pos : tuple of float
            Top-left corner in screen pixels.
        size : tuple of float
            Panel ``(width, height)`` in pixels.
        layers : list of Functional4D
            BOLD runs, plotted in order.
        cursor : ndarray or None
            Cursor in world millimeters.
        time_s : float
            Current time in seconds.
        """
        if self._implot_context is None:
            self._implot_context = implot.create_context()
        implot.set_current_context(self._implot_context)
        cursor_key = None if cursor is None else tuple(np.round(cursor, 3))
        if cursor_key != self._last_cursor:
            self._last_cursor = cursor_key
            self._fit_view = True

        imgui.set_next_window_pos(pos, imgui.Cond_.always)
        imgui.set_next_window_size(size, imgui.Cond_.always)
        imgui.push_style_color(imgui.Col_.window_bg, THEME["background"])
        imgui.push_style_var(imgui.StyleVar_.window_padding, (12, 6))
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.begin("Time Series##skyline_timeseries", None, flags)
        samples = [
            (layer, self.series(layer, cursor) if cursor is not None else None)
            for layer in layers
        ]
        self._render_header(layers, time_s)
        if not self.collapsed:
            self._render_toolbar(layers, samples, cursor)
            imgui.spacing()
            imgui.text_colored(
                THEME["text"], self._cursor_readout(samples, cursor, time_s)
            )
            self._render_plot(samples, cursor, time_s)
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color()

    def _icon_button(self, icon, tooltip, *, active=False, enabled=True):
        """Draw a clickable icon and return whether it was clicked."""
        if not enabled:
            color = THEME["shadow"]
        else:
            color = THEME["primary"] if active else THEME["secondary"]
        imgui.text_colored(color, icon)
        hovered = imgui.is_item_hovered()
        if hovered:
            imgui.set_tooltip(tooltip)
            if enabled:
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
        return enabled and imgui.is_item_clicked()

    def _render_header(self, layers, time_s):
        """Draw the title, transport controls and current time."""
        fa = icons_fontawesome_6
        imgui.align_text_to_frame_padding()
        imgui.text_colored(THEME["primary"], f"{fa.ICON_FA_CHART_LINE}  TIME SERIES")
        imgui.same_line(0, 24)
        reference = layers[0] if layers else None
        tr = reference.tr if reference is not None else 1.0
        if self._icon_button(fa.ICON_FA_BACKWARD_STEP, "Previous volume"):
            self._time_callback(max(time_s - tr, 0.0))
        imgui.same_line(0, 12)
        play_icon = fa.ICON_FA_PAUSE if self.playing else fa.ICON_FA_PLAY
        if self._icon_button(play_icon, "Pause" if self.playing else "Play"):
            self.playing = not self.playing
            self._last_step = time.perf_counter()
        imgui.same_line(0, 12)
        if self._icon_button(fa.ICON_FA_FORWARD_STEP, "Next volume"):
            self._time_callback(time_s + tr)
        imgui.same_line(0, 16)
        imgui.push_item_width(120)
        imgui.begin_group()
        changed, fps = thin_slider(
            "fps", self.fps, 1, 30, value_type="int", text_format=".0f", width=150
        )
        imgui.end_group()
        imgui.pop_item_width()
        if changed:
            self.fps = int(fps)
        imgui.same_line(0, 16)
        if reference is not None:
            frame = int(round(time_s / tr))
            imgui.text_colored(
                THEME["secondary"],
                f"t {time_s:6.1f} s   vol {frame + 1}/{reference.n_frames}",
            )
        arrow = fa.ICON_FA_ANGLE_UP if self.collapsed else fa.ICON_FA_ANGLE_DOWN
        imgui.same_line(imgui.get_window_width() - 24)
        if self._icon_button(arrow, "Expand" if self.collapsed else "Collapse"):
            self.collapsed = not self.collapsed

    @staticmethod
    def _cursor_readout(samples, cursor, time_s):
        """Return the cursor position, voxel and value text."""
        icon = icons_fontawesome_6.ICON_FA_LOCATION_CROSSHAIRS
        if cursor is None:
            return f"{icon}  no voxel picked"
        text = f"{icon}  " + "x {:.1f}   y {:.1f}   z {:.1f} mm".format(*cursor)
        for layer, sample in samples:
            if sample is not None:
                frame = min(int(round(time_s / layer.tr)), layer.n_frames - 1)
                text += f"   voxel {sample['voxel']}   {sample['raw'][frame]:.4g}"
                break
        return text

    def _render_toolbar(self, layers, samples, cursor):
        """Draw scaling, detrend, fit, spectrum, seed and export controls."""
        fa = icons_fontawesome_6
        imgui.spacing()
        label = next(
            k for k, v in _NORMALIZATION_LABELS.items() if v == self.normalization
        )
        imgui.begin_group()
        changed, label = segmented_switch(
            "Scale", list(_NORMALIZATION_LABELS), label, width=240, height=22
        )
        imgui.end_group()
        if changed:
            self.normalization = _NORMALIZATION_LABELS[label]
            self._fit_view = True
        imgui.same_line(0, 20)
        changed, self.detrend = toggle_button(self.detrend, label="Detrend")
        if changed:
            self._fit_view = True
        imgui.same_line(0, 20)
        _, self.show_fit = toggle_button(self.show_fit, label="Model fit")
        imgui.same_line(0, 20)
        changed, self.spectrum = toggle_button(self.spectrum, label="Spectrum")
        if changed:
            self._fit_view = True
        has_sample = any(sample is not None for _, sample in samples)
        imgui.same_line(0, 28)
        if self._icon_button(
            f"{fa.ICON_FA_BULLSEYE} Seed map",
            "Correlate every voxel with the time series under the cursor",
            enabled=has_sample,
        ):
            for layer, sample in samples:
                if sample is not None:
                    self._seed_callback(layer)
                    break
        imgui.same_line(0, 20)
        if self._icon_button(
            f"{fa.ICON_FA_DOWNLOAD} CSV",
            "Save the plotted time series",
            enabled=has_sample,
        ):
            render_file_dialog(
                title="Save time series",
                name="CSV Files (*.csv)",
                extensions="*.csv",
                multiselect=False,
                callback=lambda filenames=None, **_: self._export(
                    filenames, samples, cursor
                ),
                dialog_type="save",
                file_name="timeseries.csv",
                type="viz",
            )

    def _render_plot(self, samples, cursor, time_s):
        """Draw the time-course or spectrum plot, or a hint without data."""
        valid = [(layer, sample) for layer, sample in samples if sample is not None]
        if not valid:
            self._render_hint(
                "Click a slice to show the BOLD time series under it"
                if cursor is None
                else "The cursor is outside every BOLD run"
            )
            return
        implot.push_style_color(implot.Col_.frame_bg, (0, 0, 0, 0))
        implot.push_style_color(implot.Col_.plot_bg, (0.07, 0.07, 0.07, 1.0))
        implot.push_style_color(implot.Col_.plot_border, THEME["shadow"])
        implot.push_style_color(implot.Col_.legend_bg, (0.1, 0.1, 0.1, 0.85))
        implot.push_style_color(implot.Col_.axis_text, THEME["text"])
        implot.push_style_color(implot.Col_.axis_grid, (1, 1, 1, 0.06))
        if self._fit_view:
            implot.set_next_axes_to_fit()
            self._fit_view = False
        plot_flags = implot.Flags_.no_title | implot.Flags_.no_menus
        if implot.begin_plot("##bold_plot", (-1, -1), plot_flags):
            if self.spectrum:
                implot.setup_axes("frequency (Hz)", "power")
            else:
                implot.setup_axes("time (s)", _Y_LABELS[self.normalization])
            implot.setup_legend(implot.Location_.north_east)
            if self.spectrum:
                self._plot_spectra(valid)
            else:
                self._shade_events(valid[0][0])
                self._plot_series(valid)
                changed, new_time, *_ = implot.drag_line_x(
                    0, time_s, THEME["primary"], 1.5
                )
                if changed:
                    self._time_callback(max(float(new_time), 0.0))
                if implot.is_plot_hovered() and imgui.is_mouse_double_clicked(
                    imgui.MouseButton_.left
                ):
                    self._time_callback(max(implot.get_plot_mouse_pos().x, 0.0))
            implot.end_plot()
        implot.pop_style_color(6)

    @staticmethod
    def _render_hint(message):
        """Center an icon and ``message`` in the remaining panel area."""
        icon = icons_fontawesome_6.ICON_FA_LOCATION_CROSSHAIRS
        text = f"{icon}   {message}"
        available = imgui.get_content_region_avail()
        text_size = imgui.calc_text_size(text)
        cursor = imgui.get_cursor_pos()
        imgui.set_cursor_pos(
            (
                cursor.x + max((available.x - text_size.x) * 0.5, 0.0),
                cursor.y + max((available.y - text_size.y) * 0.5, 0.0),
            )
        )
        imgui.text_colored(THEME["text"], text)

    def _plot_series(self, valid):
        """Draw each run's normalized series and, when available, its fit."""
        for index, (layer, sample) in enumerate(valid):
            color = LINE_COLORS[index % len(LINE_COLORS)]
            label = _legend_label(layer.name)
            spec = implot.Spec(line_color=color, line_weight=2.0)
            implot.plot_line(label, layer.times, sample["values"], spec=spec)
            if "fit" in sample:
                fit_spec = implot.Spec(line_color=THEME["primary"], line_weight=1.5)
                implot.plot_line(
                    f"{label} task fit##fit{index}",
                    layer.times,
                    sample["fit"],
                    spec=fit_spec,
                )

    def _plot_spectra(self, valid):
        """Draw each run's power spectrum."""
        for index, (layer, sample) in enumerate(valid):
            freqs, power = power_spectrum(sample["values"], layer.tr)
            color = LINE_COLORS[index % len(LINE_COLORS)]
            spec = implot.Spec(line_color=color, line_weight=2.0)
            implot.plot_line(_legend_label(layer.name), freqs, power, spec=spec)

    @staticmethod
    def _shade_events(layer):
        """Shade each task event of ``layer`` behind the series."""
        if not layer.events:
            return
        limits = implot.get_plot_limits()
        draw_list = implot.get_plot_draw_list()
        implot.push_plot_clip_rect()
        for index, rows in enumerate(layer.events.values()):
            r, g, b, _ = CONDITION_COLORS[index % len(CONDITION_COLORS)]
            fill = imgui.get_color_u32((r, g, b, 0.09))
            edge = imgui.get_color_u32((r, g, b, 0.35))
            for onset, duration in rows:
                top_left = implot.plot_to_pixels(onset, limits.y.max)
                bottom_right = implot.plot_to_pixels(
                    onset + max(duration, layer.tr * 0.25), limits.y.min
                )
                draw_list.add_rect_filled(top_left, bottom_right, fill)
                draw_list.add_line(top_left, (top_left.x, bottom_right.y), edge, 1.0)
        implot.pop_plot_clip_rect()

    def _export(self, filenames, samples, cursor):
        """Write the plotted series of every run under the cursor to CSV."""
        if not filenames:
            return
        path = filenames if isinstance(filenames, str) else filenames[0]
        columns, headers = [], []
        for layer, sample in samples:
            if sample is None:
                continue
            columns += [layer.times, sample["raw"], sample["values"]]
            headers += [
                f"{layer.name}_time_s",
                f"{layer.name}_raw",
                f"{layer.name}_{self.normalization}",
            ]
        if not columns:
            return
        length = max(column.size for column in columns)
        table = np.full((length, len(columns)), np.nan)
        for index, column in enumerate(columns):
            table[: column.size, index] = column
        location = "cursor_mm=" + ",".join(f"{v:.2f}" for v in cursor)
        np.savetxt(
            path,
            table,
            delimiter=",",
            header=",".join(headers),
            comments=f"# {location}\n",
        )
        logger.info(f"Saved time series to {path}")
