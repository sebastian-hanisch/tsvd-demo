"""Kern: jede Einheit an der 3 x 3-Mini-Instanz von Hand (tests/conftest.py) und an Eigenschaften der geschlossenen Formeln."""

import math

import numpy as np
import pytest

import tsv_model as model
from inverse_generator import SplitMix64, grid, operator, phantom


def _parts(a, x):
    u, sv, vt = model.decompose(a)
    return u, sv, vt, vt @ x


def test_mini_tsvd_filter_shape(mini):
    assert model.tsvd_filter(3, 2).tolist() == [1.0, 1.0, 0.0]
    assert model.tsvd_filter(3, 0).tolist() == [0.0, 0.0, 0.0] and model.tsvd_filter(3, 3).tolist() == [1.0, 1.0, 1.0]
    assert model.tsvd_filter(3, 7).tolist() == [1.0, 1.0, 1.0] and model.tsvd_filter(3, -2).tolist() == [0.0, 0.0, 0.0]


def test_mini_solutions_by_hand(mini):
    a, x, b = mini
    u, sv, vt, c = _parts(a, x)
    k2 = model.solve_filter(u, sv, vt, b[None, :], model.tsvd_filter(3, 2))[0]
    k1 = model.solve_filter(u, sv, vt, b[None, :], model.tsvd_filter(3, 1))[0]
    k3 = model.solve_filter(u, sv, vt, b[None, :], model.tsvd_filter(3, 3))[0]
    assert k2 == pytest.approx([4.0 / 3.0] * 3, abs=1e-12)
    assert k1 == pytest.approx(k2, abs=1e-12)               # die zweite Komponente der Wahrheit ist null
    assert k3 == pytest.approx(x, abs=1e-12)
    assert float(model.rel_error(k2[None, :], x)[0]) == pytest.approx((math.sqrt(6.0) / 3.0) / math.sqrt(6.0), abs=1e-12)


def test_mini_bias_variance_by_hand(mini):
    a, x, b = mini
    u, sv, vt, c = _parts(a, x)
    n = 3
    for k, bias_norm, var in ((2, math.sqrt(6.0) / 3.0, 5.0), (1, math.sqrt(6.0) / 3.0, 1.0), (3, 0.0, 105.0), (0, math.sqrt(6.0), 0.0)):
        b2, v = model.bias_variance(sv, c, np.zeros(n), 1.0, model.tsvd_filter(n, k))
        assert math.sqrt(b2) == pytest.approx(bias_norm, abs=1e-12) and v == pytest.approx(var, abs=1e-9), k
    assert model.expected_error(sv, c, np.zeros(n), 0.0, model.tsvd_filter(n, 2), float(np.linalg.norm(x))) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_mini_best_rank_depends_on_the_noise(mini):
    a, x, b = mini
    u, sv, vt, c = _parts(a, x)
    xn = float(np.linalg.norm(x))
    assert model.best_rank(sv, c, np.zeros(3), 1e-4, xn)[0] == 3            # kaum Rauschen: alles behalten
    k, err = model.best_rank(sv, c, np.zeros(3), 0.5, xn)
    assert k in (1, 2) and err < 1.0                                         # viel Rauschen: schneiden (k = 1 und k = 2 haben denselben Bias)
    assert model.best_rank(sv, c, np.zeros(3), 1e3, xn)[0] == 0              # sehr viel Rauschen: nichts behalten


def test_mini_tikhonov_filter_matches_stueck_2(mini):
    a, *_ = mini
    _, sv, _ = model.decompose(a)
    assert model.tikhonov_filter(sv, 0.1) == pytest.approx([1.0 / 1.01, 0.25 / 0.26, 0.5], abs=1e-12)
    assert model.tikhonov_filter(sv, 0.0) == pytest.approx([1.0, 1.0, 1.0])
    assert model.degrees_of_freedom(model.tikhonov_filter(sv, 0.1)) == pytest.approx(1.0 / 1.01 + 0.25 / 0.26 + 0.5, abs=1e-12)


def test_degrees_of_freedom_of_the_cut_is_k(mini):
    assert [model.degrees_of_freedom(model.tsvd_filter(3, k)) for k in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_mini_best_filter_is_the_wiener_like_shrinkage(mini):
    a, x, b = mini
    u, sv, vt, c = _parts(a, x)
    f, err = model.best_filter(sv, c, np.zeros(3), 0.1, float(np.linalg.norm(x)))
    expected = np.clip(c * c / (c * c + 0.01 / sv ** 2), 0.0, 1.0)
    assert f == pytest.approx(expected, abs=1e-12) and np.all(f >= 0.0) and np.all(f <= 1.0)
    assert err <= model.best_rank(sv, c, np.zeros(3), 0.1, float(np.linalg.norm(x)))[1] + 1e-12


def test_best_filter_beats_both_families_on_the_standard_instance():
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt, c = _parts(a, x)
    xn = float(np.linalg.norm(x))
    sigma = 3.4e-4
    _, e_f = model.best_filter(sv, c, np.zeros(n), sigma, xn)
    assert e_f < model.best_rank(sv, c, np.zeros(n), sigma, xn)[1] and e_f < model.best_lambda(sv, c, np.zeros(n), sigma, xn)[1]


def test_rank_curve_bias_falls_and_std_rises_with_k():
    n = 48
    a = operator(n, 0.03)
    x = phantom(grid(n))
    u, sv, vt, c = _parts(a, x)
    cv = model.rank_curve(sv, c, np.zeros(n), 3.4e-4, float(np.linalg.norm(x)))
    assert len(cv["k"]) == n + 1 and cv["total"][0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(cv["bias"]) <= 1e-12) and np.all(np.diff(cv["std"]) >= -1e-12)
    assert cv["total"] == pytest.approx(np.hypot(cv["bias"], cv["std"]), rel=1e-12)


def test_best_lambda_beats_every_grid_point():
    n = 48
    a = operator(n, 0.03)
    x = phantom(grid(n))
    u, sv, vt, c = _parts(a, x)
    xn = float(np.linalg.norm(x))
    lam, err = model.best_lambda(sv, c, np.zeros(n), 3.4e-4, xn)
    grid_err = [model.expected_error(sv, c, np.zeros(n), 3.4e-4, model.tikhonov_filter(sv, l), xn) for l in np.logspace(-9, 1, 300)]
    assert err <= min(grid_err) + 1e-12 and 1e-4 < lam < 1e-1


def test_integration_matrices_by_hand():
    assert model.integration_matrix(3, 1) == pytest.approx(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]) / 3.0)
    assert model.integration_matrix(3, 2) == pytest.approx(np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]]) / 9.0)


def test_singular_values_of_the_integration_matrices_fall_algebraically():
    n = 48
    s1 = model.decompose(model.integration_matrix(n, 1))[1]
    s2 = model.decompose(model.integration_matrix(n, 2))[1]
    assert s1[0] / s1[-1] < 100.0 and s2[0] / s2[-1] > 1e3
    assert s1[4] * 5 == pytest.approx(s1[19] * 20, rel=0.3)       # ungefähr sigma_i ~ 1/i


def test_the_closed_form_matches_a_small_monte_carlo():
    n, w, sigma, k = 32, 0.04, 2e-4, 22
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    e = SplitMix64(1).normal(n * 3000).reshape(3000, n) * sigma
    sol = model.solve_filter(u, sv, vt, (a @ x)[None, :] + e, model.tsvd_filter(n, k))
    sim = math.sqrt(float(np.mean(model.rel_error(sol, x) ** 2)))
    assert sim == pytest.approx(model.expected_error(sv, vt @ x, np.zeros(n), sigma, model.tsvd_filter(n, k), float(np.linalg.norm(x))), rel=0.03)
