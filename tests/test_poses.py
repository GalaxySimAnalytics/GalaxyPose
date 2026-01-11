import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from galpos.poses import calculate_face_on_matrix, Orientation


def test_calculate_face_on_matrix_aligns_direction_to_z():
    d = np.array([0.2, -0.3, 0.93])
    d = d / np.linalg.norm(d)

    Rm = calculate_face_on_matrix(d)
    out = Rm @ d

    assert out.shape == (3,)
    assert np.allclose(out, [0.0, 0.0, 1.0], atol=1e-10)

    I = Rm @ Rm.T
    assert np.allclose(I, np.eye(3), atol=1e-10)


def test_calculate_face_on_matrix_zero_vector_raises():
    with pytest.raises(ValueError, match="non-zero"):
        calculate_face_on_matrix(np.array([0.0, 0.0, 0.0]))


def test_orientation_from_angmom_shapes_and_oob_nan():
    t = np.array([0.0, 1.0, 2.0])
    L = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 1.0, 0.0]])

    o = Orientation(t, angular_momentum=L)

    R = o(1.2)
    assert R.shape == (3, 3)

    R2 = o([0.5, 1.5])
    assert R2.shape == (2, 3, 3)

    R_oob = o(-1.0, extrapolate=False)
    assert np.isnan(R_oob).all()


def test_orientation_from_rotations_identity_is_constant():
    t = np.array([0.0, 1.0, 2.0])
    Rm = np.repeat(np.eye(3)[None, :, :], len(t), axis=0)

    o = Orientation(t, rotations=Rm)

    for tq in [0.0, 0.3, 1.7, 2.0]:
        R = o(tq)
        assert np.allclose(R, np.eye(3), atol=1e-12)


def test_orientation_sorts_times_and_reorders_samples():
    t_unsorted = np.array([2.0, 0.0, 1.0])
    angles_unsorted = t_unsorted.copy()
    R_unsorted = Rotation.from_euler("z", angles_unsorted).as_matrix()

    o = Orientation(t_unsorted, rotations=R_unsorted)

    expected = Rotation.from_euler("z", 1.0).as_matrix()
    got = o(1.0)
    assert np.allclose(got, expected, atol=1e-8)