import numpy as np
import pytest

from alm.core.cashflow import CashFlow
from alm.core.curve import YieldCurve
from alm.core.risk import (
    convexity,
    effective_duration,
    macaulay_duration,
    modified_duration,
    present_value_at_yield,
)


def test_pv_at_yield_matches_formula():
    # PV at a flat yield follows sum of CF_i * exp(-y * t_i).
    cf = CashFlow(times=[1, 2], amounts=[100, 100])
    y = 0.05
    expected = 100 * np.exp(-0.05 * 1) + 100 * np.exp(-0.05 * 2)
    assert present_value_at_yield(cf, y) == pytest.approx(expected, rel=1e-12)


def test_zero_coupon_duration_equals_maturity():
    # A single payment at time T has Macaulay duration exactly T.
    cf = CashFlow(times=[7], amounts=[1000])
    # Independent of the yield level.
    assert macaulay_duration(cf, 0.03) == pytest.approx(7.0, rel=1e-12)
    assert macaulay_duration(cf, 0.08) == pytest.approx(7.0, rel=1e-12)


def test_coupon_bond_duration_less_than_maturity():
    # A coupon-paying bond recovers value earlier, so D < maturity.
    # 3-year bond: coupons at t=1,2 and principal+coupon at t=3.
    cf = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])
    d = macaulay_duration(cf, 0.04)
    assert 1.0 < d < 3.0


def test_modified_equals_macaulay_under_continuous_compounding():
    # Under continuous compounding, D_mod == D_mac exactly.
    cf = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])
    y = 0.04
    assert modified_duration(cf, y) == pytest.approx(
        macaulay_duration(cf, y), rel=1e-12
    )


def test_zero_coupon_convexity_equals_maturity_squared():
    # For a single payment at T, convexity equals T^2.
    cf = CashFlow(times=[5], amounts=[1000])
    assert convexity(cf, 0.03) == pytest.approx(25.0, rel=1e-12)


def test_convexity_greater_than_duration_squared_for_spread_flows():
    # Var(t) = E[t^2] - (E[t])^2 >= 0
    # E[t^2] >= (E[t])^2
    cf = CashFlow(times=[1, 5], amounts=[100, 100])
    y = 0.03
    d = macaulay_duration(cf, y)
    c = convexity(cf, y)
    assert c > d**2


def test_duration_via_numerical_derivative():
    # D_mod = -(1/PV) * dPV/dy
    cf = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])
    y = 0.04
    h = 1e-6
    pv_up = present_value_at_yield(cf, y + h)
    pv_down = present_value_at_yield(cf, y - h)
    pv = present_value_at_yield(cf, y)
    numerical_dmod = -(pv_up - pv_down) / (2 * h) / pv
    assert modified_duration(cf, y) == pytest.approx(numerical_dmod, rel=1e-6)


def test_convexity_via_numerical_second_derivative():
    # Same idea for convexity, using a numerical second derivative of PV(y).
    cf = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])
    y = 0.04
    h = 1e-4
    pv_up = present_value_at_yield(cf, y + h)
    pv_down = present_value_at_yield(cf, y - h)
    pv = present_value_at_yield(cf, y)
    numerical_conv = (pv_up - 2 * pv + pv_down) / (h**2) / pv
    assert convexity(cf, y) == pytest.approx(numerical_conv, rel=1e-4)


def test_effective_duration_matches_modified_on_flat_curve():
    # On a flat curve, shifting the curve is the same as changing y,
    # so effective duration should match modified duration.
    curve = YieldCurve(tenors=[1, 5], rates=[0.04, 0.04])
    cf = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])
    assert effective_duration(cf, curve) == pytest.approx(
        modified_duration(cf, 0.04), rel=1e-6
    )


def test_effective_duration_zero_coupon():
    # A single payment at T has effective duration T, whatever the curve shape.
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.03])
    cf = CashFlow(times=[3], amounts=[1000])
    assert effective_duration(cf, curve) == pytest.approx(3.0, rel=1e-6)


def test_duration_raises_on_zero_pv():
    # Duration is undefined when PV is zero (offsetting flows).
    cf = CashFlow(times=[1, 1], amounts=[100, -100])
    with pytest.raises(ValueError):
        macaulay_duration(cf, 0.0)


def test_convexity_raises_on_zero_pv():
    cf = CashFlow(times=[1, 1], amounts=[100, -100])
    with pytest.raises(ValueError):
        convexity(cf, 0.0)


def test_effective_duration_raises_on_zero_pv():
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.03])
    cf = CashFlow(times=[1, 1], amounts=[100, -100])
    with pytest.raises(ValueError):
        effective_duration(cf, curve)
