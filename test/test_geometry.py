from calogeom import closest_approach_point

import numpy as np

def test_cap1():
    line1 = (
        np.array([0., 0., 0.]),
        np.array([1., 1., 0.])
    )

    line2 = (
        np.array([0., 0., 0.]),
        np.array([-1., 1., 0.])
    )

    assert np.allclose([0., 0., 0.], closest_approach_point(line1, line2))


def test_cap2():
    line1 = (
        np.array([0., 0., 0.]),
        np.array([1., 0., 1.])
    )

    line2 = (
        np.array([0., 0., 0.]),
        np.array([-1., 0., 1.])
    )

    assert np.allclose([0., 0., 0.], closest_approach_point(line1, line2))


def test_cap3():
    line1 = (
        np.array([0., 0., 0.]),
        np.array([0., 1., 1.])
    )

    line2 = (
        np.array([0., 0., 0.]),
        np.array([0., -1., 1.])
    )

    assert np.allclose([0., 0., 0.], closest_approach_point(line1, line2))
