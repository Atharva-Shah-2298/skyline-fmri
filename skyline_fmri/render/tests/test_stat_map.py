import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
if not has_fury:
    pytest.skip("Requires fury>=2.1.0", allow_module_level=True)
else:
    from skyline_fmri.render.stat_map import (
        StatMap3D,
        create_stat_map_visualization,
        default_threshold,
        stat_lut,
    )

AFFINE = np.eye(4)


def _stat():
    stat = np.zeros((12, 12, 12), dtype=np.float32)
    stat[2:5, 2:5, 2:5] = 6.0  # 27-voxel positive cluster
    stat[8, 8, 8] = 8.0  # single-voxel positive cluster
    stat[8:10, 2:4, 2:4] = -5.0  # 8-voxel negative cluster
    return stat


def _texture(viz):
    return np.swapaxes(np.asarray(viz.actor.children[0].geometry.grid.data), 0, 2)


def _lut_value(lut, value, vmax):
    index = int((value + vmax) / (2 * vmax) * lut.shape[0])
    return lut[min(index, lut.shape[0] - 1)]


def test_stat_lut_hides_values_below_threshold_and_unshown_signs():
    lut = stat_lut("Cold-Hot", 2.0, 6.0, "both")

    assert _lut_value(lut, 1.9, 6.0)[3] == 0
    assert _lut_value(lut, -1.9, 6.0)[3] == 0
    assert _lut_value(lut, 2.1, 6.0)[3] == 1
    assert _lut_value(lut, -2.1, 6.0)[3] == 1
    positive = _lut_value(lut, 5.9, 6.0)
    negative = _lut_value(lut, -5.9, 6.0)
    assert positive[0] > positive[2]  # hot: red/yellow
    assert negative[2] > negative[0]  # cold: blue/cyan

    positive_only = stat_lut("Cold-Hot", 2.0, 6.0, "positive")
    assert _lut_value(positive_only, -4.0, 6.0)[3] == 0
    assert _lut_value(positive_only, 4.0, 6.0)[3] == 1


def test_default_threshold_depends_on_map_strength():
    assert default_threshold(12.0) == pytest.approx(3.1)
    assert default_threshold(0.9) == pytest.approx(0.36)


def test_stat_map_blends_above_opaque_images_and_lists_clusters():
    viz = StatMap3D("zstat1.nii.gz", _stat(), affine=AFFINE, threshold=3.0)

    assert viz.viz_type == "stat_map"
    assert viz.actor.render_order > 0
    for child in viz.actor.children:
        assert child.material.alpha_mode == "blend"
        assert child.material.depth_write is False
        npt.assert_allclose(child.material.clim, (-viz.vmax, viz.vmax))
    assert [c["size"] for c in viz.clusters] == [1, 27, 8]
    npt.assert_allclose(viz.clusters[0]["peak_world"], [8, 8, 8])


def test_update_threshold_relabels_and_uploads_the_filtered_map():
    viz = StatMap3D("zstat1.nii.gz", _stat(), affine=AFFINE, threshold=3.0)

    viz.update_threshold(min_cluster_size=5, sign="positive")

    assert [c["size"] for c in viz.clusters] == [27]
    texture = _texture(viz)
    assert texture[8, 8, 8] == 0  # small cluster removed
    assert texture[9, 3, 3] == 0  # negative tail hidden
    assert texture[3, 3, 3] == pytest.approx(6.0)


def test_set_map_keeps_settings_and_rejects_new_shapes():
    viz = StatMap3D("r.nii.gz", _stat(), affine=AFFINE, threshold=3.0)
    replacement = np.zeros((12, 12, 12), dtype=np.float32)
    replacement[5, 5, 5] = 4.0

    viz.set_map(replacement)

    assert viz.threshold == 3.0
    assert [c["peak_index"] for c in viz.clusters] == [(5, 5, 5)]
    assert _texture(viz)[5, 5, 5] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="shape"):
        viz.set_map(np.zeros((3, 3, 3)))


def test_stat_map_rejects_non_3d_maps_and_bad_options():
    with pytest.raises(ValueError, match="3D"):
        StatMap3D("x", np.zeros((4, 4, 4, 2)), affine=AFFINE)
    with pytest.raises(ValueError, match="sign"):
        StatMap3D("x", _stat(), affine=AFFINE, sign="up")
    with pytest.raises(ValueError, match="colormap"):
        StatMap3D("x", _stat(), affine=AFFINE, colormap="Gray")


def test_create_stat_map_visualization_names_by_index():
    viz = create_stat_map_visualization((_stat(), AFFINE), 3)

    assert viz.path == "StatMap_3"
    with pytest.raises(ValueError, match="tuple"):
        create_stat_map_visualization(_stat(), 0)
