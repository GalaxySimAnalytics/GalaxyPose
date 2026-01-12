import numpy as np
import pytest

from galpos import GalaxyPoseTrajectory


def test_galaxyposetrajectory_no_orientation_call_and_final_state():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])

    g = GalaxyPoseTrajectory(t, pos, vel)

    p, v, R = g(0.5)
    assert p.shape == (3,)
    assert v.shape == (3,)
    assert R is None

    p2, v2, R2 = g([0.5, 1.5])
    assert p2.shape == (2, 3)
    assert v2.shape == (2, 3)
    assert R2 is None

    fp, fv, fR = g.final_state()
    assert np.allclose(fp, pos[-1])
    assert np.allclose(fv, vel[-1])
    assert fR is None


def test_galaxyposetrajectory_with_angmom_orientation():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    L = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 1.0, 0.0]])

    g = GalaxyPoseTrajectory(t, pos, vel, angular_momentum=L)

    p, v, R = g(1.2)
    assert p.shape == (3,)
    assert v.shape == (3,)
    assert R.shape == (3, 3)
    assert np.isfinite(R).all()

    fp, fv, fR = g.final_state()
    assert fR.shape == (3, 3)

    p,v,R = g([0.5, 1.5])
    assert p.shape == (2, 3)
    assert v.shape == (2, 3)
    assert R.shape == (2, 3, 3)
    a = g.get_acceleration(1.0)  # test acceleration call
    assert a.shape == (3,)


def test_galaxyposetrajectory_periodic_wrap():
    t = np.array([0.0, 1.0, 2.0])
    L = 10.0
    pos = np.array([[9.5, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])

    g = GalaxyPoseTrajectory(t, pos, vel, box_size=L)

    p_unwrapped, _, _ = g(0.5, wrap=False)
    p_wrapped, _, _ = g(0.5, wrap=True)

    assert p_unwrapped[0] > 9.0
    assert 0.0 <= p_wrapped[0] < L
