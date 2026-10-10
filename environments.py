"""
The five supply--demand environments used in the Monte Carlo experiment.

Each environment exposes:
  P_S(Q), P_D(Q)            — schedule functions
  P_star, Q_star             — competitive equilibrium
  values(N), costs(N)        — agent reservation arrays for population N
  name                       — string id

Linear envs are parameterised by (a, b, c, d) with P_S = a + bQ, P_D = c - dQ.
The nonlinear env uses logistic supply and quadratic-decay demand.
"""
import numpy as np


N_POP = 5000  # population size per side


# ---------- Linear envs ----------

class LinearEnv:
    def __init__(self, name, a, b, c, d, P_star, Q_star_hint=None):
        self.name = name
        self.a = a; self.b = b; self.c = c; self.d = d
        self.P_star = P_star
        if Q_star_hint is not None:
            self.Q_star = Q_star_hint
        elif (b + d) > 0:
            self.Q_star = (c - a) / (b + d)
        else:
            self.Q_star = np.nan

    def P_S(self, Q): return self.a + self.b * Q
    def P_D(self, Q): return self.c - self.d * Q

    def values_costs(self, N=N_POP):
        q = np.arange(N) + 0.5
        return self.P_D(q), self.P_S(q)


# ---------- Nonlinear env ----------

class NonlinearEnv:
    """
    Logistic supply, quadratic-decay demand.
        P_S(Q) = 0.5 + 2.0 / (1 + exp(-0.0015*(Q-3000)))
        P_D(Q) = 3.5 * (1 - (Q/6000)^2)
    Equilibrium found by Newton-Raphson (P*=2.0615, Q*=3846.61).
    """
    name = "nonlinear"
    P_star = 2.061458
    Q_star = 3846.6131

    @staticmethod
    def P_S(Q): return 0.5 + 2.0 / (1.0 + np.exp(-0.0015 * (Q - 3000)))

    @staticmethod
    def P_D(Q): return 3.5 * (1.0 - (Q / 6000.0) ** 2)

    @classmethod
    def values_costs(cls, N=N_POP):
        q = np.arange(N) + 0.5
        return cls.P_D(q), cls.P_S(q)


# ---------- Registry ----------

ENVIRONMENTS = {
    "symmetric":   LinearEnv("symmetric",  a=0.20, b=0.00040, c=4.00, d=0.00040, P_star=2.10, Q_star_hint=4750),
    "asymmetric":  LinearEnv("asymmetric", a=0.20, b=0.00032, c=4.00, d=0.00048, P_star=1.72, Q_star_hint=4750),
    "flat_supply": LinearEnv("flat_supply", a=1.50, b=0.00000, c=5.00, d=0.00100, P_star=1.50, Q_star_hint=3500),
    "flat_demand": LinearEnv("flat_demand", a=0.50, b=0.00050, c=2.80, d=0.00000, P_star=2.80, Q_star_hint=4600),
    "nonlinear":   NonlinearEnv(),
}


# ---------- IC regimes ----------

ICS = {
    "shortage": dict(phi_b=0.50, phi_s=1.05),
    "surplus":  dict(phi_b=0.95, phi_s=1.50),
}


if __name__ == "__main__":
    # Quick sanity check
    for name, env in ENVIRONMENTS.items():
        print(f"{name:15s}  P*={env.P_star:.4f}  Q*={env.Q_star:.2f}")
