import numpy as np


class YieldCurve:
    def __init__(self, tenors, rates):
        tenors = np.asarray(tenors, dtype=float)
        rates = np.asarray(rates, dtype=float)

        if tenors.ndim != 1 or rates.ndim != 1:
            raise ValueError("tenors and rates must be one-dimensional")
        if tenors.shape != rates.shape:
            raise ValueError("tenors and rates must have the same length")
        if tenors.size < 2:
            raise ValueError("at least two sample points are required")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")

        self.tenors = tenors
        self.rates = rates

    def zero_rate(self, t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.tenors, self.rates)

    def discount_factor(self, t):
        t = np.asarray(t, dtype=float)
        r = self.zero_rate(t)
        return np.exp(-r * t)

    def forward_rate(self, t1, t2):
        t1 = float(t1)
        t2 = float(t2)
        if t2 <= t1:
            raise ValueError("t2 must be strictly greater than t1")

        r1 = self.zero_rate(t1)
        r2 = self.zero_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)
