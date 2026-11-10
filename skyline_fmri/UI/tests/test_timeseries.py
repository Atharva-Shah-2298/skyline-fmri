import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
if not has_fury:
    pytest.skip("Requires fury>=2.1.0", allow_module_level=True)
else:
    from skyline_fmri.UI.timeseries import (
        EXPANDED_HEIGHT,
        HEADER_HEIGHT,
        _NORMALIZATION_LABELS,
        TimeseriesPanel,
    )
    from skyline_fmri.render.functional import Functional4D

EVENTS = {"motor": np.array([[4.0, 4.0], [16.0, 4.0]])}


def _run():
    rng = np.random.default_rng(0)
    data = 100 + rng.normal(0, 1, (5, 5, 5, 30)).astype(np.float32)
    return Functional4D("bold.nii.gz", data, affine=np.eye(4), tr=1.0, events=EVENTS)


def _panel(times=None):
    return TimeseriesPanel(
        time_callback=(times.append if times is not None else lambda t: None),
        seed_callback=lambda viz: None,
    )


def test_scale_labels_survive_the_switch_title_casing():
    # segmented_switch returns ``option.title()``; a label that changes under
    # title() made the panel raise and blank the whole UI.
    for label in _NORMALIZATION_LABELS:
        assert label.title() == label


@pytest.mark.parametrize("normalization", ["raw", "percent", "zscore"])
@pytest.mark.parametrize("spectrum", [False, True])
def test_panel_draws_every_view(ui, normalization, spectrum):
    panel = _panel()
    panel.normalization = normalization
    panel.spectrum = spectrum
    run = _run()

    for cursor in (None, np.array([2.0, 2.0, 2.0]), np.array([50.0, 0, 0])):
        for _ in range(2):
            ui.frame(
                lambda c=cursor: panel.render(
                    (0, 0), (900, EXPANDED_HEIGHT), layers=[run], cursor=c, time_s=3.0
                )
            )


def test_series_matches_the_voxel_and_fits_the_task():
    panel = _panel()
    run = _run()

    sample = panel.series(run, [2.0, 2.0, 2.0])

    assert sample["voxel"] == (2, 2, 2)
    npt.assert_array_equal(sample["raw"], run.bold[2, 2, 2])
    assert sample["values"].mean() == pytest.approx(0.0, abs=1e-9)  # percent change
    assert sample["fit"].shape == (30,)
    panel.show_fit = False
    assert "fit" not in panel.series(run, [2.0, 2.0, 2.0])
    assert panel.series(run, [50.0, 0, 0]) is None


def test_playback_steps_at_the_requested_rate_and_collapses():
    panel = _panel()
    panel.fps = 10

    assert not panel.step_due(100.0)
    panel.playing = True
    assert panel.step_due(100.0)
    assert not panel.step_due(100.05)
    assert panel.step_due(100.11)

    assert panel.height == EXPANDED_HEIGHT
    panel.collapsed = True
    assert panel.height == HEADER_HEIGHT
