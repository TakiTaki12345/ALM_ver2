# ALM_ver2
Asset-liability management toolkit for banks and life insurers.

## Conventions

- Rates are decimals (0.02 = 2%), continuously compounded.
- Shocks are in basis points (100 bp = 0.01).
- Time is in years.

## Modules

### `alm.core.curve`

| Function | Description |
| --- | --- |
| `YieldCurve(tenors, rates)` | Zero-coupon yield curve. Zero rates are interpolated linearly and held flat outside the tenors. |
| `.zero_rate(t)` | Zero rate at time `t`. |
| `.discount_factor(t)` | Discount factor `exp(-r(t) * t)`. |
| `.forward_rate(t1, t2)` | Forward rate for the period from `t1` to `t2`. |
| `.shift(shocks_bp)` | New curve with a shock added: one number for a parallel shift, or one per tenor. |

### `alm.core.cashflow`

| Function | Description |
| --- | --- |
| `CashFlow(times, amounts)` | A stream of cash flows, sorted by time. Negative amounts are outflows. |
| `.present_value(curve)` | Present value under a yield curve. |
| `.merge(other)` | Combine two streams into one, e.g. assets and liabilities. |
| `.bucket(edges)` | Total amount in each time band, e.g. `edges=[0, 1, 5, np.inf]`. |

### `alm.core.risk`

| Function | Description |
| --- | --- |
| `macaulay_duration(cashflow, y)` | PV-weighted average time to cash flow, at a flat yield `y`. |
| `modified_duration(cashflow, y)` | Sensitivity of PV to `y`; equals Macaulay duration under continuous compounding. |
| `convexity(cashflow, y)` | Second-order sensitivity of PV to `y`. |
| `effective_duration(cashflow, curve, shift_bp=1)` | Duration from shifting the whole curve down and up by `shift_bp`. Works on any curve shape. |

### `alm.core.mortality`

| Function | Description |
| --- | --- |
| `LifeTable(q, start_age=0)` | Life table from annual mortality rates `q_x`. |
| `LifeTable.from_lx(lx, start_age=0)` | Build from numbers living `l_x`. |
| `LifeTable.sult()` | Default table: the SOA Standard Ultimate Life Table, ages 20 to 120. |
| `.q_x(age)`, `.p_x(age)` | Probability of dying / surviving within one year. |
| `.survival(age, t, assumption="udd")` | Probability of surviving `t` years. Fractional years use `"udd"` or `"constant_force"`. |
| `.deferred_death_prob(age, t, assumption="udd")` | Probability of dying within `t` years. |

### `alm.scenarios.basel`

| Function | Description |
| --- | --- |
| `basel_shocks(tenors, parallel_bp, short_bp, long_bp)` | The six Basel IRRBB standard shocks, in bp at each tenor. |
| `basel_scenarios(curve, parallel_bp, short_bp, long_bp)` | `{scenario name: shocked curve}` for the six shocks. |

The six scenarios are `parallel_up`, `parallel_down`, `steepener`, `flattener`, `short_up` and `short_down`.

## Example

```python
from alm.core.cashflow import CashFlow
from alm.core.curve import YieldCurve
from alm.core.risk import effective_duration
from alm.scenarios.basel import basel_scenarios

curve = YieldCurve(tenors=[1, 5, 10], rates=[0.015, 0.02, 0.025])
bond = CashFlow(times=[1, 2, 3], amounts=[5, 5, 105])

bond.present_value(curve)          # 109.40
effective_duration(bond, curve)    # 2.87

base = bond.present_value(curve)
for name, shocked in basel_scenarios(curve, 200, 300, 150).items():
    print(name, bond.present_value(shocked) - base)
# parallel_up -6.09, parallel_down 6.46, steepener 1.37, ...
```

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python -m pytest
```