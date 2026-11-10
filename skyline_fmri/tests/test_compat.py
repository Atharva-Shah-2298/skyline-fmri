import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
if not has_fury:
    pytest.skip("Requires fury>=2.1.0", allow_module_level=True)
else:
    from skyline_fmri._compat import contour_from_roi, vertex_normals


def _ball(radius=8, size=24):
    grid = np.indices((size, size, size))
    return (np.sqrt(((grid - size // 2) ** 2).sum(0)) <= radius).astype(np.uint8)


@pytest.mark.parametrize(
    "affine", [np.eye(4), np.diag([-2.0, 2.0, 2.0, 1.0])], ids=["ras", "mirrored"]
)
def test_contour_from_roi_shades_outward_and_centers_voxels(affine):
    (actor,) = contour_from_roi(_ball(), affine=affine, iterations=20).children
    positions = actor.geometry.positions.data
    normals = actor.geometry.normals.data
    center = (affine @ [12, 12, 12, 1])[:3]
    radial = positions - center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)

    assert (radial * normals).sum(1).mean() > 0.95
    npt.assert_allclose(positions.mean(axis=0), center, atol=0.1)
    assert not actor.material.flat_shading


def test_vertex_normals_average_the_adjacent_faces():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 1]])

    normals = vertex_normals(vertices, faces)

    npt.assert_allclose(normals[2], [0, 0, 1])
    npt.assert_allclose(normals[3], [0, 1, 0])
    npt.assert_allclose(normals[0], [0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
