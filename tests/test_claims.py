"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (tsv_constants.SEED_BASE), nie über einen einzelnen Seed als Befund; die Standard-Instanz der Primäransicht ist
ausdrücklich ein einzelner Strom (Seed 35). Erwartete Fehler sind geschlossene Formeln (Erwartungswerte), Simulationswerte tragen die Toleranz ihrer Wiederholungen.
"""

import numpy as np
import pytest

import tsv_constants as C
import tsv_evaluation as ev
import tsv_model as model


@pytest.fixture(scope="module")
def rank():
    return ev.rank_experiment()


@pytest.fixture(scope="module")
def equal():
    return ev.equal_experiment()


@pytest.fixture(scope="module")
def grid_rows():
    return {r["n"]: r for r in ev.grid_experiment()}


@pytest.fixture(scope="module")
def filt():
    return ev.filter_experiment()


# ----------------------------------------------------------------------------- Standardinstanz und Presets

def test_standard_instance_expected_error_bias_std_and_dof():
    a = ev.analyse(ev.Settings())
    assert a.expected == pytest.approx(0.1144, abs=0.0005) and a.bias == pytest.approx(0.1128, abs=0.0005) and a.std == pytest.approx(0.0187, abs=0.0005)
    assert a.df == 33.0 and a.k_star == 33 and a.picard == 31 and a.cond == pytest.approx(1.26e4, rel=0.01)


def test_standard_instance_sample_errors():
    a = ev.analyse(ev.Settings())
    assert a.err_tsvd == pytest.approx(0.1153, abs=0.002) and a.err_tik == pytest.approx(0.1179, abs=0.002) and a.err_naive == pytest.approx(2.578, abs=0.02)


def test_best_rank_parameter_and_best_filter_on_the_standard_instance():
    a = ev.analyse(ev.Settings())
    assert a.err_star == pytest.approx(0.1144, abs=0.0005) and a.lam_star == pytest.approx(0.0095, rel=0.01) and a.err_tik_star == pytest.approx(0.1168, abs=0.0005)
    assert a.err_best == pytest.approx(0.0970, abs=0.0005) and a.f_best.min() >= 0.0 and a.f_best.max() <= 1.0
    assert a.err_crime_at_k == pytest.approx(0.0733, abs=0.0005)


def test_preset_numbers_in_the_help_texts():
    low = ev.analyse(ev.Settings(**C.PRESETS["Zu wenig Komponenten"]))
    high = ev.analyse(ev.Settings(**C.PRESETS["Zu viele Komponenten"]))
    fine = ev.analyse(ev.Settings(**C.PRESETS["Feineres Gitter"]))
    assert low.expected == pytest.approx(0.160, abs=0.002) and low.bias == pytest.approx(0.160, abs=0.002) and low.std < 0.002
    assert high.expected == pytest.approx(0.190, abs=0.002) and high.std == pytest.approx(0.148, abs=0.002) and high.bias == pytest.approx(0.119, abs=0.002)
    assert fine.expected == pytest.approx(0.104, abs=0.002) and fine.err_naive > 1e11 and fine.k_star == 30


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_best_rank_and_endpoints(rank):
    assert rank["k_star"] == 33 and rank["err_star"] == pytest.approx(0.1144, abs=0.0005) and rank["err_tik"] == pytest.approx(0.1168, abs=0.0005)
    assert rank["zero"] == pytest.approx(1.0, abs=1e-9) and rank["naive"] == pytest.approx(2.469, abs=0.005) and rank["df_tik"] == pytest.approx(32.9, abs=0.1)


def test_experiment_1_error_by_offset(rank):
    by = {r["offset"]: r for r in rank["rows"]}
    assert by[-10]["closed"] == pytest.approx(0.1511, abs=0.001) and by[-5]["closed"] == pytest.approx(0.1255, abs=0.001) and by[-3]["closed"] == pytest.approx(0.1195, abs=0.001)
    assert by[-1]["closed"] == pytest.approx(0.1166, abs=0.001) and by[1]["closed"] == pytest.approx(0.1147, abs=0.001) and by[2]["closed"] == pytest.approx(0.1166, abs=0.001)
    assert by[3]["closed"] == pytest.approx(0.1209, abs=0.001) and by[5]["closed"] == pytest.approx(0.1418, abs=0.001)


def test_experiment_1_cost_of_missing_the_rank(rank):
    by = {r["offset"]: r for r in rank["rows"]}
    assert by[-10]["ratio"] == pytest.approx(1.32, abs=0.01) and by[-5]["ratio"] == pytest.approx(1.10, abs=0.01) and by[-3]["ratio"] == pytest.approx(1.04, abs=0.01)
    assert by[3]["ratio"] == pytest.approx(1.06, abs=0.01) and by[5]["ratio"] == pytest.approx(1.24, abs=0.01)


def test_experiment_1_tikhonov_comparison(rank):
    by = {r["factor"]: r for r in rank["tik_rows"]}
    assert by[0.1]["ratio"] == pytest.approx(1.73, abs=0.02) and by[0.3]["ratio"] == pytest.approx(1.10, abs=0.01) and by[10.0]["ratio"] == pytest.approx(1.16, abs=0.01)


def test_experiment_1_simulation_confirms_the_closed_form(rank):
    for r in rank["rows"]:
        assert r["sim"] == pytest.approx(r["closed"], rel=0.01), r["offset"]


# ----------------------------------------------------------------------------- Experiment 2

def test_experiment_2_ratio_range(equal):
    assert equal["ratio_min"] == pytest.approx(0.905, abs=0.005) and equal["ratio_max"] == pytest.approx(1.194, abs=0.005)


def test_experiment_2_ratios_by_noise_with_gauss(equal):
    h = {r["noise"]: r["honest"]["ratio"] for r in equal["noise_rows"]}
    c = {r["noise"]: r["crime"]["ratio"] for r in equal["noise_rows"]}
    assert [round(h[k], 2) for k in sorted(h, reverse=True)] == [0.93, 1.04, 0.98, 0.98, 1.00]
    assert [round(c[k], 2) for k in sorted(c, reverse=True)] == [0.92, 1.06, 1.00, 1.02, 1.11]


def test_experiment_2_best_ranks_and_picard_numbers(equal):
    rows = {r["noise"]: r for r in equal["noise_rows"]}
    assert [rows[k]["honest"]["k"] for k in sorted(rows, reverse=True)] == [15, 28, 33, 34, 47]
    assert [rows[k]["picard"] for k in sorted(rows, reverse=True)] == [16, 26, 31, 40, 46]


def test_experiment_2_ratios_by_width(equal):
    rows = {r["w"]: r for r in equal["w_rows"]}
    assert rows[0.025]["honest"]["ratio"] == pytest.approx(0.977, abs=0.003) and rows[0.03]["honest"]["ratio"] == pytest.approx(0.979, abs=0.003)
    assert rows[0.035]["honest"]["ratio"] == pytest.approx(0.999, abs=0.003) and rows[0.04]["honest"]["ratio"] == pytest.approx(1.010, abs=0.003)
    assert rows[0.025]["crime"]["ratio"] == pytest.approx(1.043, abs=0.003) and rows[0.035]["crime"]["ratio"] == pytest.approx(1.065, abs=0.003)


def test_experiment_2_other_matrices(equal):
    by = {(r["op"], r["noise"]): r for r in equal["op_rows"]}
    assert [round(by[("integral", k)]["ratio"], 2) for k in C.EXP_NOISE_LEVELS] == [0.91, 0.95, 1.02, 1.00, 1.00]
    assert [round(by[("integral2", k)]["ratio"], 2) for k in C.EXP_NOISE_LEVELS] == [1.04, 0.92, 0.94, 1.19, 1.00]
    assert by[("integral", 1e-2)]["width"] == 40 and by[("integral2", 1e-4)]["width"] == 27 and by[("gauss", 1e-3)]["width"] == 8


# ----------------------------------------------------------------------------- Experiment 3

def test_experiment_3_best_rank_by_grid(grid_rows):
    assert [grid_rows[n]["k"] for n in (24, 32, 48, 64, 80, 96)] == [24, 30, 33, 30, 33, 30]


def test_experiment_3_errors_by_grid(grid_rows):
    assert grid_rows[32]["tsvd"] == pytest.approx(0.0414, abs=0.001) and grid_rows[32]["tik"] == pytest.approx(0.0423, abs=0.001)
    assert grid_rows[48]["tsvd"] == pytest.approx(0.1144, abs=0.001) and grid_rows[64]["tsvd"] == pytest.approx(0.0962, abs=0.001)
    assert grid_rows[80]["tsvd"] == pytest.approx(0.1019, abs=0.001) and grid_rows[96]["tsvd"] == pytest.approx(0.1026, abs=0.001)
    assert grid_rows[96]["direct"] > 1e12 and grid_rows[64]["direct"] == pytest.approx(4583, rel=0.01)


# ----------------------------------------------------------------------------- Experiment 4

def test_experiment_4_best_filter_gap(filt):
    h, c, small = filt["cases"]
    assert h["best"] == pytest.approx(0.0970, abs=0.0005) and h["tsvd"] / h["best"] - 1.0 == pytest.approx(0.18, abs=0.01)
    assert c["best"] == pytest.approx(0.0692, abs=0.0005) and c["tsvd"] / c["best"] - 1.0 == pytest.approx(0.02, abs=0.005)
    assert small["best"] == pytest.approx(0.0816, abs=0.0005) and small["tsvd"] == pytest.approx(0.0944, abs=0.0005)


def test_experiment_4_shapes(filt):
    t, k = filt["shapes"]["tsvd"], filt["shapes"]["tik"]
    assert t["peak1"] == pytest.approx(1.001, abs=0.002) and k["peak1"] == pytest.approx(0.999, abs=0.002)
    assert t["peak2"] == pytest.approx(0.674, abs=0.003) and k["peak2"] == pytest.approx(0.676, abs=0.003)
    assert t["box"] == pytest.approx(0.453, abs=0.003) and k["box"] == pytest.approx(0.462, abs=0.003)
    assert t["undershoot"] == pytest.approx(-0.026, abs=0.002) and k["undershoot"] == pytest.approx(-0.029, abs=0.002)
    assert t["rms_box"] == pytest.approx(0.0295, abs=0.001) and k["rms_box"] == pytest.approx(0.0258, abs=0.001)
    assert t["rms_box"] / k["rms_box"] == pytest.approx(1.14, abs=0.03)


def test_mini_instance_by_hand_numbers(mini):
    a, x, b = mini
    u, sv, vt = model.decompose(a)
    k2 = model.solve_filter(u, sv, vt, b[None, :], model.tsvd_filter(3, 2))[0]
    assert k2 == pytest.approx([4.0 / 3.0] * 3, abs=1e-12)
    b2, v = model.bias_variance(sv, vt @ x, np.zeros(3), 1.0, model.tsvd_filter(3, 2))
    assert b2 ** 0.5 / float(np.linalg.norm(x)) == pytest.approx(1.0 / 3.0, abs=1e-12) and v == pytest.approx(5.0, abs=1e-12)
