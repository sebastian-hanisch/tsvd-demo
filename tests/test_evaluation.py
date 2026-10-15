"""Auswertung: Formatierung, Analyse der Instanz, Verdikte (Art und Text), Form der Experimente und ihre Reproduzierbarkeit."""

import numpy as np
import pytest

import tsv_constants as C
import tsv_evaluation as ev
import tsv_model as model


# ----------------------------------------------------------------------------- Formatierung

def test_de_and_sci():
    assert ev.de(-1.234, 2) == "−1,23"
    assert ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.sci(9.96e3) == "1,0·10⁴"


def test_err_pct_digits_by_magnitude():
    assert [ev.err_pct(v) for v in (2.58, 0.1144, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,4 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]


def test_rel_text_and_noise_label():
    assert ev.rel_text(0.1144) == "11,4 % der Signalnorm" and ev.rel_text(4471.0) == "das 4,5·10³-Fache der Signalnorm"
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]


# ----------------------------------------------------------------------------- Analyse der Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_rank_n_or_more_is_the_direct_inversion_and_zero_is_the_zero_solution():
    full = ev.analyse(ev.Settings(k=48))
    over = ev.analyse(ev.Settings(k=96))
    zero = ev.analyse(ev.Settings(k=0))
    assert full.err_tsvd == pytest.approx(full.err_naive, rel=1e-9) and over.err_tsvd == pytest.approx(over.err_naive, rel=1e-9) and over.k == 48
    assert zero.err_tsvd == pytest.approx(1.0, abs=1e-12) and zero.expected == pytest.approx(1.0, abs=1e-9) and zero.df == 0.0


def test_expected_error_is_bias_and_std_combined_and_df_is_k():
    a = ev.analyse(ev.Settings())
    assert a.expected == pytest.approx(float(np.hypot(a.bias, a.std)), rel=1e-12) and a.df == pytest.approx(33.0)
    assert a.f_tsvd.sum() == pytest.approx(33.0) and set(a.f_tsvd.tolist()) == {0.0, 1.0}


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings()), ev.analyse(ev.Settings()), ev.analyse(ev.Settings(seed=36))
    assert np.array_equal(a.x_tsvd, b.x_tsvd) and not np.array_equal(a.x_tsvd, c.x_tsvd)


def test_the_curve_covers_all_ranks_and_contains_the_chosen_error():
    a = ev.analyse(ev.Settings())
    assert len(a.curve["k"]) == 49 and a.curve["total"][a.k] == pytest.approx(a.expected, rel=1e-12)
    assert int(np.argmin(a.curve["total"])) == a.k_star and a.curve["total"][a.k_star] == pytest.approx(a.err_star)


def test_the_best_filter_is_a_lower_bound_for_both_families():
    for p in C.PRESETS.values():
        a = ev.analyse(ev.Settings(**p))
        assert a.err_best <= a.err_star + 1e-12 and a.err_best <= a.err_tik_star + 1e-12
        assert a.f_best.min() >= 0.0 and a.f_best.max() <= 1.0


def test_crime_data_report_a_smaller_expected_error_than_honest_data():
    a = ev.analyse(ev.Settings(data="verbrechen"))
    assert a.err_crime_at_k < a.err_honest_at_k and a.expected == pytest.approx(a.err_crime_at_k, rel=1e-12)


# ----------------------------------------------------------------------------- Verdikte

def test_verdict_kinds_of_the_presets():
    kinds = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p)))[0] for name, p in C.PRESETS.items()}
    assert kinds == {"Standard": "success", "Zu wenig Komponenten": "warning", "Zu viele Komponenten": "warning", "Feineres Gitter": "success"}


def test_verdict_texts_name_the_dominating_part():
    low = ev.verdict_instance(ev.analyse(ev.Settings(**C.PRESETS["Zu wenig Komponenten"])))[1]
    high = ev.verdict_instance(ev.analyse(ev.Settings(**C.PRESETS["Zu viele Komponenten"])))[1]
    assert "Zu wenige Komponenten" in low and "Bias" in low and "Zu viele Komponenten" in high and "Streuung" in high


def test_crime_verdict_mentions_the_honest_error():
    a = ev.analyse(ev.Settings(data="verbrechen"))
    kind, text = ev.verdict_instance(a)
    assert kind == "info" and ev.err_pct(a.err_honest_at_k) in text


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_rank_experiment_rows_and_reproducibility():
    r1, r2 = ev.rank_experiment(reps=100), ev.rank_experiment(reps=100)
    assert r1["rows"] == r2["rows"] and r1["tik_rows"] == r2["tik_rows"] and np.array_equal(r1["curve"]["total"], r2["curve"]["total"])
    assert [r["offset"] for r in r1["rows"]] == list(C.EXP_RANK_OFFSETS)
    by = {r["offset"]: r for r in r1["rows"]}
    assert by[0]["ratio"] == pytest.approx(1.0) and all(r["ratio"] >= 1.0 - 1e-12 for r in r1["rows"])
    assert r1["zero"] == pytest.approx(1.0) and r1["naive"] > 2.0


def test_rank_experiment_simulation_confirms_the_closed_form():
    res = ev.rank_experiment()
    for r in res["rows"]:
        assert r["sim"] == pytest.approx(r["closed"], rel=0.03), r["offset"]


def test_equal_experiment_keeps_the_ratios_near_one():
    res = ev.equal_experiment()
    assert 0.85 < res["ratio_min"] <= 1.0 <= res["ratio_max"] < 1.3
    assert len(res["noise_rows"]) == len(C.EXP_NOISE_LEVELS) and len(res["op_rows"]) == len(C.OPERATORS) * len(C.EXP_NOISE_LEVELS)


def test_equal_experiment_branch_test_the_two_families_differ_somewhere():
    res = ev.equal_experiment()
    assert any(abs(r["ratio"] - 1.0) > 0.05 for r in res["op_rows"]) and any(abs(r["honest"]["ratio"] - 1.0) > 0.03 for r in res["noise_rows"])


def test_grid_experiment_rank_stays_bounded_while_direct_explodes():
    rows = {r["n"]: r for r in ev.grid_experiment()}
    assert all(28 <= rows[n]["k"] <= 36 for n in (48, 64, 80, 96)) and rows[96]["direct"] > 1e6
    assert all(rows[n]["best"] <= min(rows[n]["tsvd"], rows[n]["tik"]) + 1e-12 for n in rows)


def test_filter_experiment_cases_and_shapes():
    res = ev.filter_experiment(reps=100)
    assert [c["n"] for c in res["cases"]] == [48, 48, 24] and res["cases"][1]["honest"] is False
    h, c = res["cases"][0], res["cases"][1]
    assert h["tsvd"] / h["best"] > c["tsvd"] / c["best"]                    # mit Modellfehler ist der Abstand zum besten Filter größer
    assert res["shapes"]["tsvd"]["rms_box"] > 0 and res["shapes"]["tik"]["undershoot"] < 0


def test_operator_worlds_use_the_requested_spectrum():
    assert ev._operator_world("gauss", 48, 1e-3).cond > 1e3
    w1, w2 = ev._operator_world("integral", 48, 1e-3), ev._operator_world("integral2", 48, 1e-3)
    assert w1.cond < 100 and w2.cond > 1e3
    assert np.allclose(w1.a, model.integration_matrix(48, 1))
