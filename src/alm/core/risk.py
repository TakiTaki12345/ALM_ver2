import numpy as np

from alm.core.cashflow import CashFlow


def macaulay_duration(cashflow: CashFlow, y: float) -> float:
    # D_mac = sum_i t_i * (CF_i * exp(-y * t_i)) / PV
    t = cashflow.times
    cf = cashflow.amounts
    pv_weights = cf * np.exp(-y * t)
    pv = np.sum(pv_weights)
    if pv == 0:
        raise ValueError("Macaulay duration is undefined when PV is zero")
    return float(np.sum(t * pv_weights) / pv)


def modified_duration(cashflow: CashFlow, y: float) -> float:
    # D_mod = -(1/PV) * dPV/dy
    return macaulay_duration(cashflow, y)


def convexity(cashflow: CashFlow, y: float) -> float:
    # C = (1/PV) * d2PV/dy2 = sum_i t_i^2 * (CF_i * exp(-y * t_i)) / PV
    t = cashflow.times
    cf = cashflow.amounts
    pv_weights = cf * np.exp(-y * t)
    pv = np.sum(pv_weights)
    if pv == 0:
        raise ValueError("Convexity is undefined when PV is zero")
    return float(np.sum(t**2 * pv_weights) / pv)


def effective_duration(cashflow, curve, shift_bp=1):
    # D_eff = (PV_down - PV_up) / (2 * PV * dy)
    pv = cashflow.present_value(curve)
    if pv == 0:
        raise ValueError("Effective duration is undefined when PV is zero")
    pv_up = cashflow.present_value(curve.shift(shift_bp))
    pv_down = cashflow.present_value(curve.shift(-shift_bp))
    dy = shift_bp / 10000
    return (pv_down - pv_up) / (2 * pv * dy)
