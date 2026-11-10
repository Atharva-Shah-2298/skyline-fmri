import numpy as np
import numpy.testing as npt
import pytest

from skyline_fmri.glass import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    brain_mask,
    cluster_masks,
)


def _ball(shape=(30, 30, 30), radius=10, center=(15, 15, 15)):
    grid = np.indices(shape)
    distance = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
    return distance <= radius


def test_brain_mask_fills_dark_interior_of_a_skull_stripped_brain():
    brain = _ball().astype(np.float32) * 100
    ventricles = _ball(radius=3)
    brain[ventricles] = 5  # dark but inside the brain
    mask = brain_mask(brain)
    npt.assert_array_equal(mask, _ball())


def test_brain_mask_keeps_only_the_largest_object():
    volume = _ball().astype(np.float32) * 100
    volume[0:3, 0:3, 0:3] = 100  # isolated speck far from the brain
    mask = brain_mask(volume, closing=0)
    assert not mask[0:3, 0:3, 0:3].any()
    assert mask[15, 15, 15]


def test_brain_mask_uses_the_first_volume_of_4d_data():
    volume = np.stack([_ball(), np.zeros((30, 30, 30))], axis=-1).astype(np.float32)
    npt.assert_array_equal(brain_mask(volume), _ball())


def test_brain_mask_rejects_an_empty_volume():
    with pytest.raises(ValueError, match="no positive voxel"):
        brain_mask(np.zeros((5, 5, 5)))


def test_cluster_masks_split_tails_and_skip_empty_ones():
    display = np.zeros((4, 4, 4), dtype=np.float32)
    display[0, 0, 0] = 4.0
    display[3, 3, 3] = -5.0
    display[1, 1, 1] = np.nan
    tails = cluster_masks(display)
    assert [(sign, color) for sign, _, color in tails] == [
        ("positive", POSITIVE_COLOR),
        ("negative", NEGATIVE_COLOR),
    ]
    assert tails[0][1].sum() == 1 and tails[0][1][0, 0, 0]
    assert tails[1][1].sum() == 1 and tails[1][1][3, 3, 3]

    display[3, 3, 3] = 0
    assert [sign for sign, _, _ in cluster_masks(display)] == ["positive"]
