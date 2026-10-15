"""Kern: Filterlösungen (abgeschnittene SVD, Tikhonov, beliebiger Filter), geschlossene Bias-Varianz-Zerlegung, bester Rang, bester Parameter, bester Filter. Nur numpy.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_tsvd.py` gegen Pseudoinverse, Aufzählung, Monte Carlo und scipy.

Bezeichnungen: A = U diag(sv) V^T, x die Wahrheit, c = V^T x ihre Koeffizienten, b = A x + r + e die Daten mit Modellfehler r (bei Daten aus demselben Gitter
r = 0) und Rauschen e ~ N(0, sigma^2 I); rho_i = (U^T r)_i / sv_i. Eine Filterlösung ist x_f = V diag(f / sv) U^T b mit Filterfaktoren f (TSVD: f_i = 1 für
i <= k, sonst 0; Tikhonov: f_i = sv_i^2 / (sv_i^2 + lam^2)).
"""

from __future__ import annotations

import math

import numpy as np

_TINY = 1e-300
_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    u, sv, vt = np.linalg.svd(a)
    return u, sv, vt


def tsvd_filter(n: int, k: int) -> np.ndarray:
    """Filterfaktoren der abgeschnittenen SVD: die ersten k Komponenten bleiben, die übrigen entfallen (k wird auf [0, n] begrenzt)."""
    k = max(0, min(int(k), n))
    f = np.zeros(n)
    f[:k] = 1.0
    return f


def tikhonov_filter(sv: np.ndarray, lam: float) -> np.ndarray:
    """f_i = sv_i^2 / (sv_i^2 + lam^2)."""
    s2 = np.asarray(sv, dtype=float) ** 2
    return s2 / np.maximum(s2 + lam * lam, _TINY)


def solve_filter(u: np.ndarray, sv: np.ndarray, vt: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Lösung x_f = V diag(f / sv) U^T b für b der Form (reps, n)."""
    return ((np.atleast_2d(b) @ u) * (f / np.maximum(sv, _TINY))) @ vt


def rel_error(xhat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """||xhat - x|| / ||x|| je Zeile von xhat."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sqrt(np.sum((np.atleast_2d(xhat) - x) ** 2, axis=-1) / np.sum(x ** 2))


def bias_variance(sv: np.ndarray, c: np.ndarray, rho: np.ndarray, sigma: float, f: np.ndarray) -> tuple[float, float]:
    """(Bias^2, Varianz) einer Filterlösung als Summen der Quadrate, absolut: bias_i = f_i (c_i + rho_i) - c_i, var_i = f_i^2 sigma^2 / sv_i^2."""
    bias = f * (c + rho) - c
    var = f ** 2 * sigma ** 2 / np.maximum(sv, _TINY) ** 2
    return float(np.sum(bias ** 2)), float(np.sum(var))


def expected_error(sv, c, rho, sigma, f, x_norm) -> float:
    """Wurzel von E||x_f - x||^2, geteilt durch ||x||."""
    b2, v = bias_variance(sv, c, rho, sigma, f)
    return math.sqrt(b2 + v) / x_norm


def degrees_of_freedom(f: np.ndarray) -> float:
    """Wirksame Zahl der Parameter: Summe der Filterfaktoren (bei TSVD genau k)."""
    return float(np.sum(f))


def rank_curve(sv, c, rho, sigma, x_norm) -> dict:
    """Bias, Streuung (Wurzel der Varianz) und Gesamtfehler der abgeschnittenen SVD für alle Ränge k = 0 .. n, relativ zur Signalnorm."""
    n = len(sv)
    bias, std = np.empty(n + 1), np.empty(n + 1)
    for k in range(n + 1):
        b2, v = bias_variance(sv, c, rho, sigma, tsvd_filter(n, k))
        bias[k], std[k] = math.sqrt(b2) / x_norm, math.sqrt(v) / x_norm
    return {"k": np.arange(n + 1), "bias": bias, "std": std, "total": np.sqrt(bias ** 2 + std ** 2)}


def best_rank(sv, c, rho, sigma, x_norm) -> tuple[int, float]:
    """Bester Rang und sein erwarteter Fehler (Wahrheit bekannt, also ein Orakel); bei Gleichstand der kleinere Rang."""
    total = rank_curve(sv, c, rho, sigma, x_norm)["total"]
    k = int(np.argmin(total))
    return k, float(total[k])


def best_lambda(sv, c, rho, sigma, x_norm, lo: float = 1e-9, hi: float = 10.0, points: int = 241) -> tuple[float, float]:
    """Bester Tikhonov-Parameter und sein erwarteter Fehler (Orakel): Gitter über [lo, hi] und goldener Schnitt in der Umgebung des Minimums."""
    lams = np.logspace(math.log10(lo), math.log10(hi), points)
    errs = np.array([expected_error(sv, c, rho, sigma, tikhonov_filter(sv, float(l)), x_norm) for l in lams])
    i = int(np.argmin(errs))
    a, b = math.log10(lams[max(i - 1, 0)]), math.log10(lams[min(i + 1, points - 1)])
    f = lambda t: expected_error(sv, c, rho, sigma, tikhonov_filter(sv, 10.0 ** t), x_norm)  # noqa: E731
    x1, x2 = b - _PHI * (b - a), a + _PHI * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(60):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - _PHI * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + _PHI * (b - a)
            f2 = f(x2)
    t = 0.5 * (a + b)
    best = f(t)
    if errs[i] < best:
        return float(lams[i]), float(errs[i])
    return 10.0 ** t, float(best)


def best_filter(sv, c, rho, sigma, x_norm) -> tuple[np.ndarray, float]:
    """Bester dämpfender Diagonalfilter (0 <= f_i <= 1; Untergrenze für jedes Verfahren dieser Familie, braucht Wahrheit und Modellfehler): je Komponente
    minimiert (f (c + rho) - c)^2 + f^2 sigma^2 / sv^2 die Parabel bei f = c (c + rho) / ((c + rho)^2 + sigma^2 / sv^2); auf [0, 1] begrenzt ist das ihr
    Minimum dort. Ohne die Begrenzung würde der Filter Komponenten verstärken (f bis 7), weil er den Modellfehler kennt."""
    cc = c + rho
    denom = cc ** 2 + sigma ** 2 / np.maximum(sv, _TINY) ** 2
    f = np.clip(np.where(denom > 0.0, c * cc / np.maximum(denom, _TINY), 0.0), 0.0, 1.0)
    return f, expected_error(sv, c, rho, sigma, f, x_norm)


def integration_matrix(n: int, times: int = 1) -> np.ndarray:
    """Integrationsoperator h * (untere Dreiecksmatrix aus Einsen), `times`-fach angewendet (1: Singulärwerte ~ 1/i, 2: ~ 1/i^2)."""
    v = np.tril(np.ones((n, n))) / n
    out = np.eye(n)
    for _ in range(times):
        out = v @ out
    return out
