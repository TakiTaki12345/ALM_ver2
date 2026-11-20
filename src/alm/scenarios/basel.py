import numpy as np


def basel_shocks(tenors, parallel_bp, short_bp, long_bp):
    # The six Basel IRRBB standard shocks, in bp, at each tenor.
    # Short-rate shocks fade with maturity, long-rate shocks grow with it:
    # short(t) = short_bp * exp(-t / 4)
    # long(t)  = long_bp * (1 - exp(-t / 4))
    # basel_shocks([3.5], 100, 100, 100)["steepener"] → [25.4]
    t = np.asarray(tenors, dtype=float)
    short_shock = short_bp * np.exp(-t / 4)
    long_shock = long_bp * (1 - np.exp(-t / 4))
    return {
        "parallel_up": np.full_like(t, parallel_bp),
        "parallel_down": np.full_like(t, -parallel_bp),
        "steepener": -0.65 * short_shock + 0.9 * long_shock,
        "flattener": 0.8 * short_shock - 0.6 * long_shock,
        "short_up": short_shock,
        "short_down": -short_shock,
    }


def basel_scenarios(curve, parallel_bp, short_bp, long_bp):
    # Apply each shock to the curve's own tenors.
    # Returns {scenario name: shocked curve}; the original curve is unchanged.
    shocks = basel_shocks(curve.tenors, parallel_bp, short_bp, long_bp)
    return {name: curve.shift(shock) for name, shock in shocks.items()}
