import numpy as np
import numpy.testing as npt
import pytest

from skyline_fmri.tracts import (
    TractIndex,
    streamlines_through,
    tract_cluster_overlap,
    tract_key,
    tract_label,
)


def _line(start, stop, n=20):
    return np.linspace(start, stop, n)


def _labels():
    """Two cubic clusters: label 1 around the origin, label 2 around x=20."""
    labels = np.zeros((40, 10, 10), dtype=np.int32)
    labels[8:12, 3:7, 3:7] = 1
    labels[28:32, 3:7, 3:7] = 2
    return labels, np.array(
        [[1.0, 0, 0, -10], [0, 1.0, 0, -5], [0, 0, 1.0, -5], [0, 0, 0, 1]]
    )


def test_tract_label_names_atlas_bundles():
    assert tract_key("/data/bundles/AF_L.trk") == "AF_L"
    assert tract_label("AF_L.trk") == "Arcuate fasciculus (left)"
    assert tract_label("CST_R.trx") == "Corticospinal tract (right)"
    assert tract_label("CC_ForcepsMajor.trk") == "Forceps major"
    assert tract_label("my_bundle.tck") == "my_bundle"


def test_streamlines_through_flags_crossing_streamlines():
    labels, affine = _labels()
    streamlines = [
        _line((-10, 0, 0), (25, 0, 0)),  # crosses both clusters
        _line((-10, 4, 4), (25, 4, 4)),  # misses both
        _line((18, 0, 0), (22, 0, 0)),  # inside cluster 2 only
    ]

    npt.assert_array_equal(
        streamlines_through(streamlines, labels, affine), [True, False, True]
    )
    npt.assert_array_equal(
        streamlines_through(streamlines, labels, affine, label=1), [True, False, False]
    )


def test_tract_cluster_overlap_counts_per_cluster_and_tract():
    labels, affine = _labels()
    tracts = {
        "AF_L.trk": [_line((-10, 0, 0), (25, 0, 0)), _line((-10, 4, 4), (25, 4, 4))],
        "CST_L.trk": [_line((18, 0, 0), (22, 0, 0))] * 3,
        "UF_L.trk": [_line((-10, -4, -4), (-5, -4, -4))],
    }

    per_cluster, overall = tract_cluster_overlap(labels, affine, tracts)

    assert overall == [("CST_L.trk", 3, 1.0), ("AF_L.trk", 1, 0.5)]
    assert per_cluster[1] == [("AF_L.trk", 1, 0.5)]
    assert per_cluster[2] == [("CST_L.trk", 3, 1.0), ("AF_L.trk", 1, 0.5)]


def test_tract_index_finds_tracts_within_the_radius():
    index = TractIndex(
        {
            "AF_L.trk": [_line((0, 0, 0), (10, 0, 0))],
            "CST_L.trk": [_line((0, 20, 0), (0, 30, 0))],
        }
    )

    assert index.tracts_at([5.0, 1.0, 0.0]) == ["AF_L.trk"]
    assert index.tracts_at([0.0, 17.5, 0.0], radius=3.0) == ["CST_L.trk"]
    assert index.tracts_at([50.0, 50.0, 50.0]) == []


@pytest.mark.parametrize("empty", [[], [np.zeros((0, 3))]])
def test_empty_tracts_are_ignored(empty):
    labels, affine = _labels()

    assert tract_cluster_overlap(labels, affine, {"x": empty}) == ({}, [])
    assert TractIndex({"x": empty}).tracts_at([0, 0, 0]) == []
