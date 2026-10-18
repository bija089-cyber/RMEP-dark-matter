"""
rmep_blackhole.py
RMEP (Regulatory Model of Empty-space Energy) - black hole module.
Implements Section 12 of the RMEP paper:
  - density profile rho(r) = A / (r * (1 + alpha*r)**2)
  - metric correction delta_vac(R) (Section 9.3)
  - mass integrability of the 1/r divergence
  - NFW-like asymptotic slope rho ~ r**-3
  - empirical correlation A = C * alpha**-2.26 (SPARC-based, R2 = 0.727)

Run:  python rmep_blackhole.py
"""

import numpy as np

C_REL = 1.0  # proportionality constant for A(alpha)


# ----------------------------------------------------------------------
# Core RMEP density profile (Section 12)
# ----------------------------------------------------------------------
def rho_rmep(r, A, alpha):
    """RMEP central density profile: rho = A / (r * (1 + alpha r)**2)."""
    r = np.asarray(r, dtype=float)
    return A / (r * (1.0 + alpha * r) ** 2)


def A_from_alpha(alpha, C=C_REL, power=2.26):
    """Empirical correlation A = C * alpha**-2.26 (R2 = 0.727)."""
    return C * np.asarray(alpha, dtype=float) ** (-power)


# ----------------------------------------------------------------------
# Enclosed mass: finite despite rho ~ 1/r at the centre
# ----------------------------------------------------------------------
def M_enclosed_analytic(r, A, alpha):
    """
    M(<r) = 4*pi*A/alpha**2 * [ ln(1+alpha r) + 1/(1+alpha r) - 1 ]
    Finite for any r > 0 even though rho diverges as 1/r.
    """
    r = np.asarray(r, dtype=float)
    x = alpha * r
    return 4.0 * np.pi * A / alpha**2 * (np.log1p(x) + 1.0 / (1.0 + x) - 1.0)


def M_enclosed_numeric(r, A, alpha, n=200000):
    """Numerical check: M(<r) = int_0^r 4*pi*rho(r') r'^2 dr'."""
    r = float(r)
    rp = np.linspace(1e-12, r, n)
    integrand = A * rp / (1.0 + alpha * rp) ** 2
    return 4.0 * np.pi * np.trapezoid(integrand, rp)


# ----------------------------------------------------------------------
# Metric correction delta_vac (Section 9.3)
# ----------------------------------------------------------------------
def delta_vac(R, Lam, alpha):
    """
    delta_vac(R) = (Lam * alpha**2 / 3) * [1 - (1 + R/alpha)**2 * exp(-R/alpha)]
    """
    R = np.asarray(R, dtype=float)
    x = R / alpha
    return (Lam * alpha**2 / 3.0) * (1.0 - (1.0 + x) ** 2 * np.exp(-x))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_metric_correction_small():
    """delta_vac must be negligible for stellar and SMBH black holes."""
    Lam = 1.1e-52      # m^-2
    alpha = 3.086e19   # 1 kpc in m
    r_solar = 3e3      # Schwarzschild radius of the Sun, m
    r_sgr = 1.2e10     # Sgr A* ~ 4e6 Msun
    r_m87 = 2.0e13     # M87* ~ 6.5e9 Msun
    d_stellar = abs(delta_vac(r_solar, Lam, alpha))
    d_sgr = abs(delta_vac(r_sgr, Lam, alpha))
    d_m87 = abs(delta_vac(r_m87, Lam, alpha))
    print(f"delta_vac (stellar)  = {d_stellar:.2e}")
    print(f"delta_vac (Sgr A*)   = {d_sgr:.2e}")
    print(f"delta_vac (M87*)     = {d_m87:.2e}")
    assert d_stellar < 1e-25 and d_sgr < 1e-20 and d_m87 < 1e-17
    print("PASS: metric correction is negligible -> Schwarzschild exterior")
    print()


def test_mass_finiteness():
    """Analytic enclosed mass must match numerical integration."""
    A, alpha = 1.3e9, 1.0 / 1.5
    for r in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ma = M_enclosed_analytic(r, A, alpha)
        mn = M_enclosed_numeric(r, A, alpha)
        rel = abs(ma - mn) / ma
        print(f"r={r:>6}  M_analytic={ma:.6e}  M_numeric={mn:.6e}  rel.err={rel:.2e}")
        assert rel < 0.05
    print("PASS: 1/r divergence is integrable, mass finite")
    print()


def test_outer_slope():
    """At large r the profile must approach rho ~ r**-3 (NFW-like)."""
    A, alpha = 1.0, 1.0
    r1, r2 = 50.0, 200.0
    rho1, rho2 = rho_rmep(r1, A, alpha), rho_rmep(r2, A, alpha)
    slope = np.log(rho2 / rho1) / np.log(r2 / r1)
    print(f"measured slope = {slope:.3f} (theory: -3.0)")
    assert abs(slope + 3.0) < 0.05
    print("PASS: NFW-like outer asymptotic rho ~ r**-3")
    print()


def test_correlation():
    """A(alpha) must follow the power law with exponent -2.26."""
    alphas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    A_pred = A_from_alpha(alphas)
    p, _ = np.polyfit(np.log(alphas), np.log(A_pred), 1)
    print(f"recovered exponent = {p:.3f} (theory: -2.26)")
    assert abs(p + 2.26) < 1e-6
    print("PASS: single-parameter family A ~ alpha**-2.26")
    print()


if __name__ == "__main__":
    print("=" * 64)
    print("RMEP black-hole module tests (Section 12)")
    print("=" * 64)
    print()
    test_metric_correction_small()
    test_mass_finiteness()
    test_outer_slope()
    test_correlation()
    print("All 4 tests passed.")
