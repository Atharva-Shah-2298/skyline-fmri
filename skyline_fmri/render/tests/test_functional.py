import logging

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
if not has_fury:
    pytest.skip("Requires fury>=2.1.0", allow_module_level=True)
else:
    from skyline_fmri.render.functional import (
        Functional4D,
        create_functional_visualization,
    )

AFFINE = np.array([[2.0, 0, 0, -10], [0, 2.0, 0, -10], [0, 0, 2.0, -10], [0, 0, 0, 1]])


def _bold(n_frames=6):
    rng = np.random.default_rng(0)
    data = 100 + rng.random((6, 7, 8, n_frames)).astype(np.float32)
    data[..., :] += np.arange(n_frames, dtype=np.float32) * 10
    return data


def _texture(viz):
    return np.swapaxes(np.asarray(viz.actor.children[0].geometry.grid.data), 0, 2)


def test_set_frame_swaps_the_displayed_volume_and_reports_time():
    reported = []
    viz = Functional4D(
        "bold.nii.gz",
        _bold(),
        affine=AFFINE,
        tr=2.0,
        time_callback=lambda v, t: reported.append(t),
    )

    viz.set_frame(3)

    assert viz.frame == 3
    npt.assert_allclose(_texture(viz), viz.bold[..., 3])
    assert reported == [6.0]
    viz.set_frame(99)
    assert viz.frame == 5


def test_set_time_selects_the_nearest_volume_without_reporting():
    reported = []
    viz = Functional4D(
        "bold.nii.gz",
        _bold(),
        affine=AFFINE,
        tr=2.0,
        time_callback=lambda v, t: reported.append(t),
    )

    viz.set_time(6.9)

    assert viz.frame == 3
    assert reported == []


def test_display_modes_show_summaries_and_refit_limits():
    viz = Functional4D("bold.nii.gz", _bold(), affine=AFFINE, tr=1.0)
    frame_limits = viz.value_range

    viz.set_display_mode("tSNR")
    tsnr = viz.bold.mean(-1) / viz.bold.std(-1)
    npt.assert_allclose(_texture(viz), tsnr, rtol=1e-5)
    assert viz.value_range != frame_limits

    viz.set_frame(2)
    npt.assert_allclose(_texture(viz), tsnr, rtol=1e-5)  # summaries ignore time
    viz.set_display_mode("Volume")
    npt.assert_allclose(_texture(viz), viz.bold[..., 2])
    with pytest.raises(ValueError, match="mode"):
        viz.set_display_mode("Max")


def test_timeseries_at_maps_world_points_through_the_affine():
    viz = Functional4D("bold.nii.gz", _bold(), affine=AFFINE, tr=1.0)

    voxel, series = viz.timeseries_at([-6.0, -4.0, -2.0])

    assert voxel == (2, 3, 4)
    npt.assert_array_equal(series, viz.bold[2, 3, 4])
    assert viz.timeseries_at([100.0, 0, 0]) is None


def test_missing_tr_defaults_to_one_second_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        viz = Functional4D("bold.nii.gz", _bold(), affine=AFFINE)

    assert viz.tr == 1.0
    assert "repetition time" in caplog.text
    npt.assert_allclose(viz.times, np.arange(6))


def test_functional_reads_nibabel_images_and_rejects_3d_data():
    img = nib.Nifti1Image(_bold(), AFFINE)

    viz = Functional4D("bold.nii.gz", img, tr=2.0)

    npt.assert_allclose(viz.affine, AFFINE)
    assert viz.bold.dtype == np.float32
    assert viz.viz_type == "functional"
    with pytest.raises(ValueError, match="4D"):
        Functional4D("x", _bold()[..., 0], affine=AFFINE)


def test_load_events_file_attaches_conditions_and_logs_failures(tmp_path, caplog):
    viz = Functional4D("bold.nii.gz", _bold(), affine=AFFINE, tr=1.0)
    good = tmp_path / "events.tsv"
    good.write_text("onset\tduration\ttrial_type\n0\t2\tmotor\n")
    bad = tmp_path / "bad.tsv"
    bad.write_text("start\n1\n")

    assert viz.load_events_file(str(good))
    npt.assert_array_equal(viz.events["motor"], [[0, 2]])
    with caplog.at_level(logging.ERROR):
        assert not viz.load_events_file(str(bad))
    assert list(viz.events) == ["motor"]


def test_create_functional_visualization_prefers_the_explicit_tr():
    viz = create_functional_visualization((_bold(), AFFINE, "run.nii.gz", 3.0), 0)
    assert viz.tr == 3.0

    viz = create_functional_visualization(
        (_bold(), AFFINE, "run.nii.gz", 3.0), 0, tr=0.5
    )
    assert viz.tr == 0.5
    assert viz.path == "run.nii.gz"


def test_percent_change_mode_plays_as_a_transparent_overlay():
    data = np.zeros((6, 7, 8, 40), dtype=np.float32)
    data[1:5, 1:6, 1:7] = 100.0
    data[1:4, 1:4, 1:4, 20:25] += 3.0  # a 3% response in a small blob
    viz = Functional4D("bold.nii.gz", data, affine=AFFINE, tr=2.0)
    assert not viz.is_overlay

    viz.set_display_mode("Change")
    viz.set_frame(22)

    assert viz.is_overlay
    texture = _texture(viz)
    npt.assert_allclose(texture, viz.percent[..., 22])
    assert texture[2, 2, 2] > 1.5
    assert texture[0, 0, 0] == 0  # outside the brain
    assert 0 <= viz.change_threshold < viz.change_max
    for child in viz.actor.children:
        assert child.material.alpha_mode == "blend"
        npt.assert_allclose(child.material.clim, (-viz.change_max, viz.change_max))
    assert viz.actor.render_order > 0

    viz.set_display_mode("Volume")
    npt.assert_allclose(_texture(viz), data[..., 22])
    assert viz.actor.render_order == 0
    assert all(c.material.alpha_mode == "bayer" for c in viz.actor.children)
