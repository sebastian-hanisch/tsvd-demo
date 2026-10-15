"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen gegen einen anderen Rechenweg (Pseudoinverse, Aufzählung, Spur, scipy, Monte Carlo).

Jeder Test läuft in unter 10 s. Fehlt scipy im Test-venv, überspringen sich die Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import tsv_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom

scipy = pytest.importorskip("scipy")
from scipy import optimize  # noqa: E402


@pytest.mark.parametrize("n,w,k", [(24, 0.02, 10), (48, 0.03, 15), (48, 0.03, 33), (64, 0.03, 30), (96, 0.04, 28)])
def test_truncated_solution_matches_the_pseudoinverse_of_the_rank_k_matrix(n, w, k):
    a = operator(n, w)
    b = draw_data(n, w, 1e-3, SplitMix64(n), crime=False)[0]
    u, sv, vt = model.decompose(a)
    ours = model.solve_filter(u, sv, vt, b[None, :], model.tsvd_filter(n, k))[0]
    ak = (u[:, :k] * sv[:k]) @ vt[:k]
    ref = np.linalg.pinv(ak, rcond=0.5 * sv[k - 1] / sv[0]) @ b          # rcond zwischen sigma_k und dem numerischen Nullbereich
    assert np.linalg.norm(ours - ref) / np.linalg.norm(ours) < 1e-8


def test_the_rank_k_matrix_is_the_best_approximation_in_the_spectral_norm():
    """Eckart-Young: ||A - A_k||_2 = sigma_{k+1}; gegen die Spektralnorm von numpy."""
    n, w, k = 32, 0.03, 12
    a = operator(n, w)
    u, sv, vt = model.decompose(a)
    ak = (u[:, :k] * sv[:k]) @ vt[:k]
    assert float(np.linalg.norm(a - ak, 2)) == pytest.approx(float(sv[k]), rel=1e-9)


@pytest.mark.parametrize("honest", [False, True])
def test_best_rank_matches_an_enumeration_over_all_ranks(honest):
    n, w, noise = 48, 0.03, 1e-3
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c = vt @ x
    rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv if honest else np.zeros(n)
    sigma = noise_sigma(n, w, noise)
    xn = float(np.linalg.norm(x))
    errs = []
    for k in range(n + 1):
        bias = np.concatenate([rho[:k], -c[k:]])                     # unabhängig aufgebaut: behalten gibt rho, weggeworfen gibt -c
        var = sigma ** 2 * float(np.sum(1.0 / sv[:k] ** 2))
        errs.append(math.sqrt(float(np.sum(bias ** 2)) + var) / xn)
    k_ref = int(np.argmin(errs))
    k, e = model.best_rank(sv, c, rho, sigma, xn)
    assert k == k_ref and e == pytest.approx(errs[k_ref], rel=1e-12)


@pytest.mark.parametrize("honest", [False, True])
def test_expected_error_matches_a_monte_carlo(honest):
    n, w, noise, k = 48, 0.03, 1e-3, 34
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    sigma = noise_sigma(n, w, noise)
    rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv if honest else np.zeros(n)
    b = draw_data(n, w, noise, SplitMix64(11), crime=not honest, reps=3000)
    f = model.tsvd_filter(n, k)
    sim = math.sqrt(float(np.mean(model.rel_error(model.solve_filter(u, sv, vt, b, f), x) ** 2)))
    assert sim == pytest.approx(model.expected_error(sv, vt @ x, rho, sigma, f, float(np.linalg.norm(x))), rel=0.02)


def test_degrees_of_freedom_equal_the_trace_of_the_projection():
    n, k = 48, 33
    a = operator(n, 0.03)
    u, sv, vt = model.decompose(a)
    proj = vt[:k].T @ vt[:k]
    assert float(np.trace(proj)) == pytest.approx(model.degrees_of_freedom(model.tsvd_filter(n, k)), abs=1e-9)


@pytest.mark.parametrize("noise", [1e-1, 1e-2, 1e-3, 1e-4])
def test_best_filter_matches_scipy_minimization_per_component(noise):
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c = vt @ x
    rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv
    sigma = noise_sigma(n, w, noise)
    f, _ = model.best_filter(sv, c, rho, sigma, float(np.linalg.norm(x)))
    for i in (0, 10, 25, 33, 40, 47):
        res = optimize.minimize_scalar(lambda t, i=i: (t * (c[i] + rho[i]) - c[i]) ** 2 + t * t * sigma ** 2 / sv[i] ** 2, bounds=(0.0, 1.0), method="bounded",
                                       options={"xatol": 1e-12})
        assert f[i] == pytest.approx(float(res.x), abs=1e-5), i


@pytest.mark.parametrize("noise", [1e-2, 1e-3])
def test_best_lambda_matches_scipy_bounded_minimization(noise):
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c, xn = vt @ x, float(np.linalg.norm(x))
    rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv
    sigma = noise_sigma(n, w, noise)
    _, ours = model.best_lambda(sv, c, rho, sigma, xn)
    res = optimize.minimize_scalar(lambda t: model.expected_error(sv, c, rho, sigma, model.tikhonov_filter(sv, 10.0 ** t), xn), bounds=(-9.0, 1.0), method="bounded",
                                   options={"xatol": 1e-10})
    assert ours == pytest.approx(float(res.fun), rel=1e-4) and ours <= float(res.fun) + 1e-9
