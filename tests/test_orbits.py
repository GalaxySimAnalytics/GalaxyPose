import numpy as np
import pytest

from galpos.orbits import wrap_to_box, unwrap_positions, PolynomialInterpolator, Trajectory


def test_wrap_to_box_range():
    x = np.array([-0.1, 0.0, 9.9, 10.0, 10.1, 20.2])
    L = 10.0
    w = wrap_to_box(x, L)
    assert np.all((0.0 <= w) & (w < L))
    assert np.allclose(w, np.mod(x, L))


def test_unwrap_positions_continuous_minimal_image():
    L = 10.0
    pos = np.array([
        [9.5, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    unwrapped = unwrap_positions(pos, L)
    expected = np.array([
        [9.5, 0.0, 0.0],
        [10.2, 0.0, 0.0],
        [11.0, 0.0, 0.0],
    ])
    assert np.allclose(unwrapped, expected)


def test_polynomial_interpolator_cubic_satisfies_endpoint_constraints():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    dydx = np.array([0.0, 0.0])

    p = PolynomialInterpolator(x, y, dydx, extrapolate=False)

    assert np.allclose(p(0.0), 0.0)
    assert np.allclose(p(1.0), 1.0)

    dp = p.derivative()
    assert np.allclose(dp(0.0), 0.0, atol=1e-12)
    assert np.allclose(dp(1.0), 0.0, atol=1e-12)


def test_trajectory_spline_scalar_and_vector_shapes_and_values():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])

    tr = Trajectory(t, pos, vel, method="spline")

    p, v = tr(0.5)
    assert p.shape == (3,)
    assert v.shape == (3,)
    assert np.allclose(p, [0.5, 0.0, 0.0])
    assert np.allclose(v, [1.0, 0.0, 0.0])

    p2, v2 = tr([0.5, 1.5])
    assert p2.shape == (2, 3)
    assert v2.shape == (2, 3)
    assert np.allclose(p2[:, 0], [0.5, 1.5], atol=1e-12)


def test_trajectory_periodic_unwrap_and_wrap_output():
    t = np.array([0.0, 1.0, 2.0])
    L = 10.0
    pos = np.array([[9.5, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])

    tr = Trajectory(t, pos, vel, box_size=L, method="spline")

    p_unwrapped, _ = tr(0.5, wrap=False)
    p_wrapped, _ = tr(0.5, wrap=True)

    assert p_unwrapped[0] > 9.0
    assert 0.0 <= p_wrapped[0] < L


def test_trajectory_out_of_bounds_nan_when_no_extrapolate():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.ones_like(pos)

    tr = Trajectory(t, pos, vel, method="spline")

    p, v = tr(-1.0, extrapolate=False)
    assert np.isnan(p).all()
    assert np.isnan(v).all()

    p2, v2 = tr(-1.0, extrapolate=True)
    assert np.isfinite(p2).all()
    assert np.isfinite(v2).all()


def test_trajectory_pchip_when_velocities_missing():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.5, 0.0],
                    [2.0, 1.0, 0.0]])
    with pytest.warns(RuntimeWarning, match="switching method to 'pchip'"):
        tr = Trajectory(t, pos, velocities=None, method="spline")

    assert tr.method == "pchip"

    p, v = tr([0.5, 1.5])
    assert p.shape == (2, 3)
    assert v.shape == (2, 3)

    assert 0.0 < p[0, 0] < 1.0
    assert 1.0 < p[1, 0] < 2.0


def test_trajectory_switches_to_polynomial_when_accelerations_provided():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.ones_like(pos)
    acc = np.zeros_like(pos)
    with pytest.warns(RuntimeWarning, match="switching method to 'polynomial'"):
        tr = Trajectory(t, pos, velocities=vel, accelerations=acc, method="spline")
    assert tr.method == "polynomial"

    p, v = tr(0.5)
    assert p.shape == (3,)
    assert v.shape == (3,)