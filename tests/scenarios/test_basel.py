import numpy as np
import pytest

from alm.core.curve import YieldCurve
from alm.scenarios.basel import basel_scenarios, basel_shocks


def test_steepener_matches_rbi_example():
    # RBI worked example: short and long shocks of 100bp give a steepener
    # of +25.4bp at t = 3.5.
    shocks = basel_shocks([3.5], 100, 100, 100)
    assert shocks["steepener"][0] == pytest.approx(25.4, abs=0.05)


def test_parallel_shocks():
    shocks = basel_shocks([1, 5], 200, 300, 150)
    assert list(shocks["parallel_up"]) == [200.0, 200.0]
    assert list(shocks["parallel_down"]) == [-200.0, -200.0]


def test_short_shocks_fade_with_maturity():
    # short(t) = short_bp * exp(-t / 4): full size at t = 0, e^-1 of it at t = 4.
    shocks = basel_shocks([0, 4], 200, 300, 150)
    assert shocks["short_up"] == pytest.approx([300, 300 * np.exp(-1)], abs=1e-9)
    assert shocks["short_down"] == pytest.approx([-300, -300 * np.exp(-1)], abs=1e-9)


def test_flattener():
    # At t = 4: short = 100 * e^-1, long = 100 * (1 - e^-1).
    shocks = basel_shocks([4], 200, 100, 100)
    expected = 0.8 * 100 * np.exp(-1) - 0.6 * 100 * (1 - np.exp(-1))
    assert shocks["flattener"][0] == pytest.approx(expected, abs=1e-9)


def test_six_scenarios():
    shocks = basel_shocks([1, 5], 200, 300, 150)
    assert set(shocks) == {
        "parallel_up",
        "parallel_down",
        "steepener",
        "flattener",
        "short_up",
        "short_down",
    }


def test_basel_scenarios_shift_the_curve():
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.03])
    scenarios = basel_scenarios(curve, 200, 300, 150)
    # +200bp = +0.02 at every tenor.
    assert scenarios["parallel_up"].rates == pytest.approx([0.04, 0.05], abs=1e-9)
    # The original curve is unchanged.
    assert curve.rates == pytest.approx([0.02, 0.03], abs=1e-9)
