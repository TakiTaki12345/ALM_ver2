import numpy as np


class CashFlow:
    def __init__(self, times, amounts):
        times = np.asarray(times, dtype=float)
        amounts = np.asarray(amounts, dtype=float)

        # Validate on construction so every method downstream can trust the
        # data is well-formed (strict on genuine errors).
        if times.ndim != 1 or amounts.ndim != 1:
            raise ValueError("times and amounts must be one-dimensional")
        if times.shape != amounts.shape:
            raise ValueError("times and amounts must have the same length")
        if times.size == 0:
            raise ValueError("a cash flow must have at least one payment")
        if np.any(times < 0):
            raise ValueError("times must be non-negative")

        order = np.argsort(times)
        self.times = times[order]
        self.amounts = amounts[order]

    def present_value(self, curve):
        # PV = sum_i amount_i * discount_factor(time_i)
        discount_factors = curve.discount_factor(self.times)
        return float(np.sum(self.amounts * discount_factors))

    def merge(self, other):
        # Combine two cash flow streams into one, e.g. assets and liabilities.
        # bond = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])
        # deposit = CashFlow(times=[1, 2], amounts=[-50, -60])
        # net.times   → [1, 1, 2, 2, 3]
        # net.amounts → [5, -50, 5, -60, 105]
        times = np.concatenate([self.times, other.times])
        amounts = np.concatenate([self.amounts, other.amounts])
        return CashFlow(times, amounts)

    def bucket(self, edges):
        # net.bucket([0, 1.5, 2.5, np.inf])
        # [0, 1.5)   : 5 - 50 = -45
        # [1.5, 2.5) : 5 - 60 = -55
        # [2.5, ∞)   : 105
        # → [-45, -55, 105]
        edges = np.asarray(edges, dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing")

        totals = np.zeros(len(edges) - 1)
        for i in range(len(edges) - 1):
            in_band = (self.times >= edges[i]) & (self.times < edges[i + 1])
            totals[i] = np.sum(self.amounts[in_band])
        return totals
