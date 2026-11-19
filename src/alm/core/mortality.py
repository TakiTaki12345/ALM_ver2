import numpy as np


class LifeTable:
    def __init__(self, q, start_age=0):
        # q: annual mortality rates q_x for ages start_age, start_age + 1, ...
        q = np.asarray(q, dtype=float)

        if q.ndim != 1:
            raise ValueError("q must be one-dimensional")
        if q.size == 0:
            raise ValueError("a life table needs at least one rate")
        if np.any(q < 0) or np.any(q > 1):
            raise ValueError("each q_x must be a probability in [0, 1]")

        self.q = q
        self.start_age = int(start_age)

    @classmethod
    def from_lx(cls, lx, start_age=0):
        # Build from numbers living l_x: q_x = (l_x - l_{x+1}) / l_x.
        # The last l_x has no next age, so the table has one fewer age.
        lx = np.asarray(lx, dtype=float)
        if lx.ndim != 1:
            raise ValueError("lx must be one-dimensional")
        if lx.size < 2:
            raise ValueError("lx must contain at least two ages")
        if np.any(lx <= 0):
            raise ValueError("l_x must be positive")
        if np.any(np.diff(lx) > 0):
            raise ValueError("l_x must be non-increasing")

        q = (lx[:-1] - lx[1:]) / lx[:-1]
        return cls(q, start_age=start_age)

    @classmethod
    def sult(cls):
        # Default table: the SOA Standard Ultimate Life Table (SULT).
        # Makeham's law, mu_x = A + B * c^x, p_x = exp(-A - B * c^x * (c - 1) / ln(c))
        # Ages 20 to 120, annuity
        # values use the ages beyond.
        # LifeTable.sult().q_x(65) → 0.005915
        a, b, c = 0.00022, 2.7e-6, 1.124
        ages = np.arange(20, 121)
        p = np.exp(-a - b * c**ages * (c - 1) / np.log(c))
        return cls(1 - p, start_age=20)

    def q_x(self, age):
        # Probability that a life aged `age` dies within one year.
        i = int(age) - self.start_age
        if not 0 <= i < self.q.size:
            raise ValueError(f"age {age} is outside the table")
        return float(self.q[i])

    def p_x(self, age):
        # Probability that a life aged `age` survives one year.
        return 1.0 - self.q_x(age)

    def survival(self, age, t, assumption="udd"):
        # t_p_x: probability that a life aged `age` survives t years.
        # Split t into whole years n and a fraction s:  t_p_x = n_p_x * s_p_{x+n}
        # Fraction of a year:
        #   "udd"            : s_p_y = 1 - s * q_y   (deaths spread evenly)
        #   "constant_force" : s_p_y = p_y ** s
        if t < 0:
            raise ValueError("t must be non-negative")
        if assumption not in ("udd", "constant_force"):
            raise ValueError('assumption must be "udd" or "constant_force"')

        n = int(np.floor(t))
        s = t - n

        # Whole years: survive each of ages age, age + 1, ..., age + n - 1.
        whole_years = 1.0
        for k in range(n):
            whole_years *= self.p_x(age + k)
        if s == 0:
            return whole_years

        q_y = self.q_x(age + n)
        if assumption == "udd":
            fraction = 1 - s * q_y
        else:
            fraction = (1 - q_y) ** s
        return whole_years * fraction

    def deferred_death_prob(self, age, t, assumption="udd"):
        # t_q_x = 1 - t_p_x
        return 1.0 - self.survival(age, t, assumption=assumption)
