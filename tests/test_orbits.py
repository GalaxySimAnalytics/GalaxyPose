import numpy as np
import pytest

from galpos.orbits import (
    wrap_to_box,
    unwrap_positions,
    PolynomialInterpolator,
    Trajectory,
)


def test_wrap():
    # Case: inputs including negatives, equal to L,
    # and exceeding L should wrap to [0, L)
    x = np.array([-0.1, 0.0, 9.9, 10.0, 10.1, 20.2])
    L = 10.0
    w = wrap_to_box(x, L)
    assert np.all((0.0 <= w) & (w < L))
    assert np.allclose(w, np.mod(x, L))


def test_unwrap_minimal_image_3d():
    # Case: crossing periodic boundary (9.5 -> 0.2)
    # should unwrap to continuous trajectory using minimum image convention
    L = 10.0
    pos = np.array(
        [
            [9.5, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    unwrapped = unwrap_positions(pos, L)
    expected = np.array(
        [
            [9.5, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(unwrapped, expected)


def test_poly_cubic_endpoints_and_nan_oob():
    # Case: only first derivative given -> cubic;
    # endpoint values/derivatives satisfy constraints; no extrapolation returns NaN
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 5.0])
    dy = np.array([2.0, 2.0])
    p = PolynomialInterpolator(x, y, dy, extrapolate=False)

    assert np.allclose(p(0.0), 1.0)
    assert np.allclose(p(2.0), 5.0)
    dp = p.derivative()
    assert np.allclose(dp(0.0), 2.0, atol=1e-12)
    assert np.allclose(dp(2.0), 2.0, atol=1e-12)

    # 越界且extrapolate=False -> NaN
    assert np.isnan(p(-1.0))
    assert np.isnan(dp(3.0))


def test_poly_quintic_endpoints():
    # Case: second derivative given -> quintic;
    # endpoint first/second derivatives satisfy constraints
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    dy = np.array([0.0, 0.0])
    d2 = np.array([2.0, 2.0])  # constant second derivative
    p = PolynomialInterpolator(x, y, dy, d2, extrapolate=False)

    assert np.allclose(p(0.0), 0.0)
    assert np.allclose(p(1.0), 1.0)

    dp = p.derivative()
    ddp = p.derivative(2)
    assert np.allclose(dp(0.0), 0.0, atol=1e-12)
    assert np.allclose(dp(1.0), 0.0, atol=1e-12)
    assert np.allclose(ddp(0.0), 2.0, atol=1e-12)
    assert np.allclose(ddp(1.0), 2.0, atol=1e-12)

def test_poly_axis_not0_and_derivative_type():
    # Case: multi-dimensional input with axis!=0 triggers moveaxis;
    # derivative returns PolynomialInterpolator instance
    t = np.array([0.0, 1.0, 2.0])
    # shape (2, 3), sample points along axis=1
    y = np.vstack([t, 2.0 * t])
    dy = np.vstack([np.ones_like(t), 2.0 * np.ones_like(t)])

    p = PolynomialInterpolator(t, y, dy, extrapolate=False, axis=1)
    v = p(0.5)
    dv = p.derivative()(0.5)

    assert v.shape == (2,)
    assert dv.shape == (2,)
    assert np.allclose(dv, [1.0, 2.0], atol=1e-12)
    assert isinstance(p.derivative(), PolynomialInterpolator)


def test_poly_bad_lengths():
    # Case: input lengths mismatch -> raises ValueError
    # (covers _compute_coefficients validation branch)
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])          # wrong length
    dy = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        PolynomialInterpolator(x, y, dy)

    # Case: second derivative length mismatch -> raises ValueError
    y2 = np.array([0.0, 1.0, 2.0])
    d2 = np.array([0.0, 0.0])         # wrong length
    with pytest.raises(ValueError, match="does not match"):
        PolynomialInterpolator(x, y2, dy, d2ydx2=d2)


def test_traj_bad_inputs():
    # Case: passing unknown method -> raises ValueError
    t = np.array([0.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.ones_like(pos)
    with pytest.raises(ValueError, match="Unknown method"):
        Trajectory(t, pos, vel, method="nope")

    # Case: times not 1D -> raises ValueError
    with pytest.raises(ValueError, match="times must be a 1D"):
        Trajectory(np.array([[0.0, 1.0]]), pos, vel, method="spline")

    # Case: duplicate times (not strictly increasing after sorting)
    # -> raises ValueError
    t_dup = np.array([0.0, 1.0, 1.0])
    pos_dup = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0]])
    vel_dup = np.ones_like(pos_dup)
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(t_dup, pos_dup, vel_dup, method="spline")


def test_traj_sort_and_wrap_output():
    # Case: input times unordered -> internally sorted;
    # periodic box -> internally unwrap, output can choose wrap
    times = np.array([2.0, 0.0, 1.0])
    L = 10.0
    pos = np.array([[1.0, 0.0, 0.0],
                    [9.5, 0.0, 0.0],
                    [0.2, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])

    tr = Trajectory(times, pos, vel, box_size=L, method="spline")
    assert np.all(np.diff(tr.times) > 0)

    p_unwrapped, _ = tr(0.5, wrap=False)
    p_wrapped, _ = tr(0.5, wrap=True)
    assert p_unwrapped.shape == (3,)
    assert p_wrapped.shape == (3,)
    assert p_unwrapped[0] > 9.0
    assert 0.0 <= p_wrapped[0] < L


def test_traj_call_extrapolate_nan_and_finite():
    # Case: extrapolate=False returns NaN outside bounds;
    # extrapolate=True returns finite values outside bounds (covers __call__ branch)
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.ones_like(pos)

    tr = Trajectory(t, pos, vel, method="spline")

    p0, v0 = tr(-1.0, extrapolate=False)
    assert np.isnan(p0).all()
    assert np.isnan(v0).all()

    p1, v1 = tr(-1.0, extrapolate=True)
    assert np.isfinite(p1).all()
    assert np.isfinite(v1).all()

    # Case: array input returns (M, ndim)
    p2, v2 = tr([0.5, 1.5])
    assert p2.shape == (2, 3)
    assert v2.shape == (2, 3)


def test_traj_switch_to_pchip_when_no_vel():
    # Case: velocities and accelerations both missing and method not pchip
    # -> warns and switches to pchip; can return velocity/acceleration
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.5, 0.0],
                    [2.0, 1.0, 0.0]])

    with pytest.warns(RuntimeWarning, match="switching method to 'pchip'"):
        tr = Trajectory(t, pos, velocities=None, accelerations=None, method="spline")

    assert tr.method == "pchip"
    p, v = tr([0.5, 1.5])
    a = tr.get_acceleration([0.5, 1.5])
    assert p.shape == (2, 3)
    assert v.shape == (2, 3)
    assert a.shape == (2, 3)


def test_traj_switch_to_poly_when_acc_and_estimate_vel_1d():
    # Case: providing acc and requesting non-polynomial
    # -> warns and switches to polynomial; missing vel triggers "estimate velocity" branch (1D)
    t = np.array([0.0, 1.0, 2.0])
    a = np.array([2.0, 2.0, 2.0])  # 1D constant acceleration
    # x = 0.5*a*t^2
    pos = 0.5 * a * t**2

    with pytest.warns(RuntimeWarning, match="switching method to 'polynomial'"):
        tr = Trajectory(t, pos, velocities=None, accelerations=a, method="spline")

    assert tr.method == "polynomial"
    # Case: scalar input returns scalar float (covers scalar_input branch)
    p_s, v_s = tr(0.5)
    assert np.isscalar(p_s)
    assert np.isscalar(v_s)

    # Case: acceleration scalar and array input
    a_s = tr.get_acceleration(0.5)
    a_v = tr.get_acceleration([0.5, 1.5])
    assert np.isscalar(a_s)
    assert a_v.shape == (2,)
    assert np.isfinite(a_s)
    assert np.isfinite(a_v).all()


def test_traj_methods_with_velocities_and_pchip_override():
    # Case: with velocities, override spline/polynomial/pchip paths respectively;
    #       pchip will override passed velocities with interpolated derivatives (covers that branch)
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.ones_like(pos)

    tr_s = Trajectory(t, pos, vel, method="spline")
    assert tr_s.method == "spline"
    assert tr_s.get_acceleration(1.0).shape == (3,)

    tr_p = Trajectory(t, pos, vel, method="polynomial")
    assert tr_p.method == "polynomial"
    assert tr_p.get_acceleration([0.5, 1.5]).shape == (2, 3)

    vel_bad = 123.0 * np.ones_like(pos)
    tr_c = Trajectory(t, pos, vel_bad, method="pchip")
    assert tr_c.method == "pchip"
    # pchip will override self.velocities with interpolated derivatives; should not keep vel_bad
    assert not np.allclose(tr_c.velocities, vel_bad)


def test_from_orbit():
    # Case: from_orbit factory method constructs correctly and can compute acceleration
    a = np.array([2.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 2.0])
    v0 = np.array([0.5, 0.0, 0.0])
    x0 = np.array([0.0, 0.0, 0.0])

    pos = np.stack([0.5 * a * tt**2 + v0 * tt + x0 for tt in t])
    vel = np.stack([a * tt + v0 for tt in t])

    tr = Trajectory.from_orbit(pos, vel, t, method="polynomial")
    a_eval = tr.get_acceleration(0.5)
    assert a_eval.shape == (3,)
    assert np.isfinite(a_eval).all()

def test_poly_axis_quintic_moveaxis():
    # Case: axis!=0 and providing second derivative d2ydx2
    #      walking quintic (fifth-order) multi-dimensional path, and endpoint constraints on second derivative hold
    t = np.array([0.0, 1.0])
    # shape=(2, 2), time points on axis=1: two components are y=t and y=2t (linear functions with zero second derivative)
    y = np.vstack([t, 2.0 * t])
    dy = np.vstack([np.ones_like(t), 2.0 * np.ones_like(t)])
    d2 = np.zeros_like(y)

    p = PolynomialInterpolator(t, y, dy, d2ydx2=d2, extrapolate=False, axis=1)

    # Endpoint second derivative should be 0 (verifies quintic constraints + moveaxis(d2) effect)
    ddp = p.derivative(2)
    assert np.allclose(ddp(0.0), [0.0, 0.0], atol=1e-12)
    assert np.allclose(ddp(1.0), [0.0, 0.0], atol=1e-12)


def test_traj_sort_with_acc():
    # Case: times unordered and providing accelerations -> should sort positions/velocities/accelerations accordingly
    times = np.array([2.0, 0.0, 1.0])

    # Use 1D scalar trajectory for easy checking of sorting "correspondence"
    pos = np.array([20.0, 0.0, 10.0])
    vel = np.array([2.0, 0.0, 1.0])
    acc = np.array([200.0, 0.0, 100.0])

    tr = Trajectory(times, pos, velocities=vel, accelerations=acc, method="polynomial")

    assert np.allclose(tr.times, [0.0, 1.0, 2.0])
    assert np.allclose(tr.positions, [0.0, 10.0, 20.0])
    assert np.allclose(tr.velocities, [0.0, 1.0, 2.0])
    assert np.allclose(tr.accelerations, [0.0, 100.0, 200.0])


def test_traj_estimate_vel_nd_with_acc():
    # Case: accelerations provided but velocities missing, and positions are multi-dimensional (N, ndim)
    #      -> walks "estimate velocity" multi-dimensional branch dp=np.diff(pos,axis=0) and /dt[:,None];
    #         and covers vel[-1] = vel[-2] copy logic
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 6.0],
        ]
    )
    acc = np.zeros_like(pos)

    # Case: request non-polynomial method, should trigger "switching method to 'polynomial'" warning branch
    with pytest.warns(RuntimeWarning, match="switching method to 'polynomial'"):
        tr = Trajectory(t, pos, velocities=None, accelerations=acc, method="spline")

    assert tr.method == "polynomial"

    # Expected velocities:
    # v0 = (pos1-pos0)/(t1-t0) = [1,2]
    # v1 = (pos2-pos1)/(t2-t1) = [2,4]
    # v2 = v1 (last one copied)
    assert np.allclose(tr.velocities[0], [1.0, 2.0])
    assert np.allclose(tr.velocities[1], [2.0, 4.0])
    assert np.allclose(tr.velocities[2], [2.0, 4.0])

    # Add a scalar acceleration evaluation, which usually also hits the scalar return branch of get_acceleration
    a_s = tr.get_acceleration(0.5)
    assert a_s.shape == (2,)
    assert np.isfinite(a_s).all()
