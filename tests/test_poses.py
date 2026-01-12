import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from galpos.poses import (
    calculate_face_on_matrix,
    Orientation,
    quaternion_multiply,
    quaternion_inverse,
    quaternion_logarithm,
    quaternion_exponential,
)


def _is_so3(Rm: np.ndarray, atol: float = 1e-10) -> bool:
    if Rm.ndim == 2:
        return (
            np.allclose(Rm @ Rm.T, np.eye(3), atol=atol)
            and np.isclose(np.linalg.det(Rm), 1.0, atol=atol)
            and np.all(np.isfinite(Rm))
        )
    return (
        np.allclose(Rm @ np.transpose(Rm, (0, 2, 1)), np.eye(3), atol=atol)
        and np.allclose(np.linalg.det(Rm), 1.0, atol=atol)
        and np.all(np.isfinite(Rm))
    )


def test_quat_mul_inv_id():
    # Case: algebraic property of quaternion_multiply + quaternion_inverse (q * q^{-1} = I)
    q = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_quat()  # [x,y,z,w]
    q = np.array([q[3], q[0], q[1], q[2]])  # -> [w,x,y,z]
    got = quaternion_multiply(q, quaternion_inverse(q))
    assert np.allclose(got, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_quat_logexp_small():
    # Case: small-argument branch for log/exp (v_norm < 1e-10)
    qI = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion_logarithm(qI), np.zeros(4), atol=1e-12)
    assert np.allclose(quaternion_exponential(np.zeros(4)), qI, atol=1e-12)


def test_quat_logexp_roundtrip():
    # Case: non-small branch (exp of a pure quaternion -> unit quaternion; log recovers it)
    axis = np.array([1.0, 2.0, -1.0])
    axis = axis / np.linalg.norm(axis)
    theta = 0.7
    x = np.array([0.0, *(theta * axis)])  # 纯四元数 [0, v]
    q = quaternion_exponential(x)
    x2 = quaternion_logarithm(q)
    assert np.allclose(x2[0], 0.0, atol=1e-12)
    assert np.allclose(x2[1:], x[1:], atol=1e-12)


def test_face_on_ok():
    # Case: normal face-on (direction not parallel to up)
    d = np.array([0.0, 0.0, 2.0])
    Rm = calculate_face_on_matrix(d)
    out = Rm @ (d / np.linalg.norm(d))
    assert np.allclose(out, [0.0, 0.0, 1.0], atol=1e-12)
    assert _is_so3(Rm, atol=1e-12)


def test_face_on_alt_up():
    # Case: direction parallel to up -> triggers alt up branch (perp1_norm < 1e-6)
    d = np.array([0.0, 1.0, 0.0])  # parallel to default up=[0,1,0]
    Rm = calculate_face_on_matrix(d)
    out = Rm @ (d / np.linalg.norm(d))
    assert np.allclose(out, [0.0, 0.0, 1.0], atol=1e-12)
    assert _is_so3(Rm, atol=1e-12)


def test_face_on_bad_raises():
    # Case: invalid direction vector (zero vector / NaN) -> ValueError
    with pytest.raises(ValueError, match="non-zero finite 3-vector"):
        calculate_face_on_matrix(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="non-zero finite 3-vector"):
        calculate_face_on_matrix(np.array([np.nan, 0.0, 0.0]))


def test_init_times_ndim_raises():
    # Case: times is not 1D -> ValueError
    t = np.array([[0.0, 1.0]])
    Rm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    with pytest.raises(ValueError, match="1D"):
        Orientation(t, rotations=Rm)


def test_init_duplicate_raises():
    # Case: times contains duplicates (even if sort would fix) -> ValueError
    t = np.array([0.0, 1.0, 1.0])
    Rm = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        Orientation(t, rotations=Rm)


def test_init_missing_mode_raises():
    # Case: neither rotations nor angular_momentum provided -> ValueError
    with pytest.raises(ValueError, match="Either rotations or angular_momentum"):
        Orientation(np.array([0.0, 1.0, 2.0]))


def test_init_sort_rotations():
    # Case: unsorted times + rotations should be reordered accordingly
    t = np.array([2.0, 0.0, 1.0])
    ang = np.array([0.8, 0.0, 0.4])  # mapping: t=2->0.8, t=0->0.0, t=1->0.4
    Rm = Rotation.from_euler("z", ang).as_matrix()
    o = Orientation(t, rotations=Rm)

    assert np.allclose(o.times, [0.0, 1.0, 2.0], atol=0.0)

    # t=0 corresponds to angle 0.0, so rotation should be identity
    R0 = o(0.0)
    assert np.allclose(R0, np.eye(3), atol=1e-12)


def test_init_sort_angmom_and_fallback():
    # Case: unsorted times + angmom reorder + zero vector triggers fallback [0,0,1]
    t = np.array([2.0, 0.0, 1.0])
    L = np.array(
        [
            [0.0, 0.0, 1.0],  # t=2
            [0.0, 0.0, 0.0],  # t=0 -> fallback
            [0.0, 1.0, 0.0],  # t=1
        ]
    )
    o = Orientation(t, angular_momentum=L)
    assert o.use_angmom_interp is True
    assert np.allclose(o.times, [0.0, 1.0, 2.0])

    # fallback produces face-on aligning z axis -> rotation should be close to identity
    R0 = o(0.0)
    assert np.allclose(R0, np.eye(3), atol=1e-12)


def test_call_oob_nan_array_and_scalar():
    # Case: extrapolate=False and fully out-of-bounds -> returns NaN matrix (both array/scalar branches)
    t = np.array([0.0, 1.0, 2.0])
    Rm = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    o = Orientation(t, rotations=Rm)

    out_a = o(np.array([-2.0, 3.0]), extrapolate=False)
    assert out_a.shape == (2, 3, 3)
    assert np.isnan(out_a).all()

    out_s = o(-1.0, extrapolate=False)
    assert out_s.shape == (3, 3)
    assert np.isnan(out_s).all()


def test_call_extrapolate_true_ok():
    # Case: extrapolate=True allows extrapolation; returns SO(3) and finite
    t = np.array([0.0, 1.0, 2.0])
    Rm = Rotation.from_euler("z", [0.0, 0.4, 0.8]).as_matrix()
    o = Orientation(t, rotations=Rm)

    out = o(np.array([-1.0, 0.5, 3.0]), extrapolate=True)
    assert out.shape == (3, 3, 3)
    assert _is_so3(out, atol=1e-8)


def test_dir_interp_branches():
    # Case: direction interpolation covers key branches
    # - interval[0,1]: dot<0 -> neg_mask; after flipping dot~1 -> linear_mask
    # - interval[1,2]: dot~0 -> slerp_mask
    # - t=3.0: resulting direction == up -> parallel_mask (enters alt_up branch)
    t = np.array([0.0, 1.0, 2.0, 3.0])
    L = np.array(
        [
            [0.0, 0.0, 1.0],   # z
            [0.0, 0.0, -1.0],  # -z (triggers neg_mask)
            [1.0, 0.0, 0.0],   # x (about 90deg with z triggers slerp)
            [0.0, 1.0, 0.0],   # up (used to trigger parallel_mask)
        ]
    )
    o = Orientation(t, angular_momentum=L)

    # Note: the last point is at t=3.0 (endpoint), ensuring the interpolated direction is exactly up, thus triggering parallel_mask
    Rm = o(np.array([0.5, 1.5, 3.0]), extrapolate=False)
    assert Rm.shape == (3, 3, 3)
    assert _is_so3(Rm, atol=1e-10)

    # Additional assertions: at t=3.0, z-axis is up, and the basis constructed with alt_up=[1,0,0] is deterministic
    assert np.allclose(Rm[2, 2, :], [0.0, 1.0, 0.0], atol=1e-12)  # z-axis row
    assert np.allclose(Rm[2, 0, :], [0.0, 0.0, 1.0], atol=1e-12)  # x-axis row (alt_up × z)
    assert np.allclose(Rm[2, 1, :], [1.0, 0.0, 0.0], atol=1e-12)  # y-axis row (z × x)


def test_batch_slerp_neg_and_linear_and_slerp():
    # Case: _batch_slerp covers neg_mask / linear_mask / slerp_mask
    qa = np.array([[1.0, 0.0, 0.0, 0.0]])

    # neg_mask: dot < 0
    qb = np.array([[-1.0, 0.0, 0.0, 0.0]])
    out = Orientation._batch_slerp(qa, qb, np.array([0.5]))
    assert np.allclose(out[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    # slerp branch: midpoint of 90deg about x should be 45deg about x
    qb2 = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0]])
    out2 = Orientation._batch_slerp(qa, qb2, np.array([0.5]))
    expected = np.array([np.cos(np.deg2rad(22.5)), np.sin(np.deg2rad(22.5)), 0.0, 0.0])
    assert np.allclose(out2[0], expected, atol=1e-10)


def test_slerp_wrapper_scalar():
    # Case: scalar wrapper for _slerp works
    o = Orientation(np.array([0.0, 1.0]), rotations=Rotation.from_euler("x", [0.0, 0.0]).as_matrix())
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0])
    mid = o._slerp(q1, q2, 0.5)
    assert np.allclose(mid, [np.cos(np.deg2rad(22.5)), np.sin(np.deg2rad(22.5)), 0.0, 0.0], atol=1e-10)


def test_control_points_two():
    # Case: _compute_squad_control_points branch for num_points<=2
    t = np.array([0.0, 1.0])
    Rm = Rotation.from_euler("x", [0.0, 0.3]).as_matrix()
    o = Orientation(t, rotations=Rm)

    assert len(o.control_points) == 2
    assert np.allclose(o.control_points[0], o.quaternions[0], atol=1e-12)
    assert np.allclose(o.control_points[1], o.quaternions[1], atol=1e-12)


def test_control_points_many_and_squad():
    # Case: _compute_squad_control_points interior + endpoints (num_points>2)
    # Also triggers _squad_interpolation and verifies output is SO(3)
    t = np.array([0.0, 1.0, 2.0, 3.0])
    Rm = Rotation.from_euler("xyz", np.array([[0.0, 0.0, 0.0],
                                             [0.1, -0.2, 0.3],
                                             [0.2, -0.1, 0.4],
                                             [0.3,  0.0, 0.5]])).as_matrix()
    o = Orientation(t, rotations=Rm)

    assert len(o.control_points) == 4
    assert np.all(np.isfinite(np.array(o.control_points)))

    mid = o(np.array([0.3, 1.7, 2.2]), extrapolate=False)
    assert mid.shape == (3, 3, 3)
    assert _is_so3(mid, atol=1e-8)


def test_rotmat_quat_roundtrip():
    # Case: rotation matrix <-> quaternion conversion path coverage
    Rm = Rotation.from_euler("xyz", [0.2, -0.4, 0.1]).as_matrix()[None, :, :]
    q = Orientation._rotation_matrices_to_quaternions(Rm)[0]  # [w,x,y,z]
    Rm2 = Orientation._quaternion_to_rotation_matrix(q)
    assert np.allclose(Rm2, Rm[0], atol=1e-12)


def test_normalize_quats_mask():
    # Case: _normalize_quaternions mask branch (including zero quaternions should not divide by zero)
    o = Orientation(np.array([0.0, 1.0]), rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0))
    qs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # norm=0 -> mask False
            [2.0, 0.0, 0.0, 0.0],  # -> [1,0,0,0]
        ]
    )
    out = o._normalize_quaternions(qs)
    assert np.allclose(out[0], [0.0, 0.0, 0.0, 0.0], atol=0.0)
    assert np.allclose(out[1], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
