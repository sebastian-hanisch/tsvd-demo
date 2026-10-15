"""Auswertung: Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tsv_constants as C
import tsv_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def lam_text(lam: float) -> str:
    """Strafparameter: ab 0,01 mit Dezimalstellen, darunter als Zehnerpotenz."""
    if lam == 0.0:
        return "0"
    return de(lam, 3 if lam >= 0.1 else 4) if lam >= 0.01 else sci(lam)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def rel_text(err: float) -> str:
    """Fehler als Anteil der Signalnorm in Worten."""
    if not math.isfinite(err):
        return "unendlich viel"
    if err >= 10.0:
        return f"das {sci(err)}-Fache der Signalnorm"
    return f"{err_pct(err)} der Signalnorm"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


# ----------------------------------------------------------------------------- Welt

@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    k: int = C.DEFAULT_K
    seed: int = C.DEFAULT_SEED


class World:
    """Matrix, Zerlegung, Wahrheit und Modellfehler einer Instanz; Fehler beliebiger Filter in geschlossener Form."""

    def __init__(self, n: int, w: float, noise: float | None = None, sigma: float | None = None, a: np.ndarray | None = None):
        self.n, self.w = n, w
        self.a = operator(n, w) if a is None else a
        self.u, self.sv, self.vt = model.decompose(self.a)
        self.x = phantom(grid(n))
        self.g = blurred_exact(grid(n), w) if a is None else self.a @ self.x
        self.c = self.vt @ self.x
        self.rho_honest = (self.u.T @ (self.g - self.a @ self.x)) / np.maximum(self.sv, 1e-300)
        if sigma is not None:
            self.sigma = float(sigma)
        elif a is None:
            self.sigma = noise_sigma(n, w, noise)
        else:
            self.sigma = float(noise) * float(np.sqrt(np.mean(self.g ** 2)))
        self.x_norm = float(np.linalg.norm(self.x))
        self.cond = float(self.sv[0] / self.sv[-1]) if self.sv[-1] > 0 else math.inf

    def rho(self, honest: bool) -> np.ndarray:
        return self.rho_honest if honest else np.zeros(self.n)

    def err(self, f: np.ndarray, honest: bool = True) -> float:
        return model.expected_error(self.sv, self.c, self.rho(honest), self.sigma, f, self.x_norm)

    def best_k(self, honest: bool = True) -> tuple[int, float]:
        return model.best_rank(self.sv, self.c, self.rho(honest), self.sigma, self.x_norm)

    def best_lam(self, honest: bool = True) -> tuple[float, float]:
        return model.best_lambda(self.sv, self.c, self.rho(honest), self.sigma, self.x_norm)

    def best_f(self, honest: bool = True) -> tuple[np.ndarray, float]:
        return model.best_filter(self.sv, self.c, self.rho(honest), self.sigma, self.x_norm)

    def picard(self) -> int:
        return int(np.sum(np.abs(self.u.T @ self.g) > self.sigma))


# ----------------------------------------------------------------------------- Instanz

@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    g: np.ndarray
    sv: np.ndarray
    cond: float
    sigma: float
    k: int                   # wirksamer Rang (begrenzt auf n)
    b: np.ndarray
    x_naive: np.ndarray
    x_tsvd: np.ndarray
    x_tik: np.ndarray        # Tikhonov beim besten Parameter (Orakel), zum Vergleich
    err_naive: float
    err_tsvd: float          # Fehler dieser Stichprobe
    err_tik: float
    expected: float          # erwarteter Fehler der abgeschnittenen SVD zum gewählten k
    bias: float
    std: float
    df: float
    f_tsvd: np.ndarray
    f_tik: np.ndarray
    f_best: np.ndarray
    k_star: int              # bester Rang (Orakel)
    err_star: float
    lam_star: float
    err_tik_star: float
    err_best: float
    picard: int
    curve: dict
    err_honest_at_k: float
    err_crime_at_k: float
    x_norm: float
    singular: bool


def analyse(settings: Settings) -> Analysis:
    st = settings
    wd = World(st.n, st.w, st.noise)
    honest = st.data != "verbrechen"
    k = max(0, min(int(st.k), st.n))
    b = draw_data(st.n, st.w, st.noise, SplitMix64(st.seed), crime=not honest)
    rho = wd.rho(honest)
    f_ts = model.tsvd_filter(st.n, k)
    k_star, err_star = wd.best_k(honest)
    lam_star, err_tik_star = wd.best_lam(honest)
    f_tik = model.tikhonov_filter(wd.sv, lam_star)
    f_best, err_best = wd.best_f(honest)
    bias2, var = model.bias_variance(wd.sv, wd.c, rho, wd.sigma, f_ts)
    xt = model.solve_filter(wd.u, wd.sv, wd.vt, b, f_ts)[0]
    xl = model.solve_filter(wd.u, wd.sv, wd.vt, b, f_tik)[0]
    xn = model.solve_filter(wd.u, wd.sv, wd.vt, b, np.ones(st.n))[0]
    return Analysis(
        settings=st, s=grid(st.n), x=wd.x, g=wd.g, sv=wd.sv, cond=wd.cond, sigma=wd.sigma, k=k, b=b[0], x_naive=xn, x_tsvd=xt, x_tik=xl,
        err_naive=float(model.rel_error(xn, wd.x)[0]), err_tsvd=float(model.rel_error(xt, wd.x)[0]), err_tik=float(model.rel_error(xl, wd.x)[0]),
        expected=math.sqrt(bias2 + var) / wd.x_norm, bias=math.sqrt(bias2) / wd.x_norm, std=math.sqrt(var) / wd.x_norm, df=model.degrees_of_freedom(f_ts),
        f_tsvd=f_ts, f_tik=f_tik, f_best=f_best, k_star=k_star, err_star=err_star, lam_star=lam_star, err_tik_star=err_tik_star, err_best=err_best,
        picard=wd.picard(), curve=model.rank_curve(wd.sv, wd.c, rho, wd.sigma, wd.x_norm),
        err_honest_at_k=wd.err(f_ts, True), err_crime_at_k=wd.err(f_ts, False), x_norm=wd.x_norm, singular=bool(wd.cond >= C.SINGULAR_COND),
    )


def standard_sigma() -> float:
    """Absolutes Rauschen der Standardinstanz: bleibt in Experiment 3 fest, wenn das Gitter wechselt."""
    return noise_sigma(C.DEFAULT_N, C.DEFAULT_W, C.DEFAULT_NOISE)


# ----------------------------------------------------------------------------- Experiment 1: der Schnitt und seine Nachsicht

def rank_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    """Fehler bei Rängen um den besten Rang (ehrliche Daten): geschlossen und Simulation; daneben Tikhonov bei Vielfachen des besten Parameters."""
    wd = World(n, w, noise)
    k_star, err_star = wd.best_k(True)
    lam_star, err_tik = wd.best_lam(True)
    b = draw_data(n, w, noise, SplitMix64(C.SEED_BASE + 1), crime=False, reps=reps)
    rows = []
    for d in C.EXP_RANK_OFFSETS:
        k = max(0, min(k_star + d, n))
        f = model.tsvd_filter(n, k)
        closed = wd.err(f, True)
        sim = float(np.sqrt(np.mean(model.rel_error(model.solve_filter(wd.u, wd.sv, wd.vt, b, f), wd.x) ** 2)))
        rows.append({"offset": d, "k": k, "closed": closed, "sim": sim, "ratio": closed / err_star})
    tik_rows = []
    for fac in C.EXP_TIK_FACTORS:
        e = wd.err(model.tikhonov_filter(wd.sv, lam_star * fac), True)
        tik_rows.append({"factor": fac, "lam": lam_star * fac, "closed": e, "ratio": e / err_tik})
    curve = model.rank_curve(wd.sv, wd.c, wd.rho(True), wd.sigma, wd.x_norm)
    return {"k_star": k_star, "err_star": err_star, "lam_star": lam_star, "err_tik": err_tik, "rows": rows, "tik_rows": tik_rows, "curve": curve,
            "naive": float(curve["total"][n]), "zero": float(curve["total"][0]), "df_tik": model.degrees_of_freedom(model.tikhonov_filter(wd.sv, lam_star)),
            "reps": reps, "n": n}


# ----------------------------------------------------------------------------- Experiment 2: zwei Filter, ein Ergebnis

def _operator_world(op: str, n: int, noise: float) -> World:
    if op == "gauss":
        return World(n, C.DEFAULT_W, a=operator(n, C.DEFAULT_W), noise=noise)
    return World(n, 0.0, a=model.integration_matrix(n, 1 if op == "integral" else 2), noise=noise)


def _compare(wd: World, honest: bool) -> dict:
    k, e_k = wd.best_k(honest)
    lam, e_l = wd.best_lam(honest)
    _, e_f = wd.best_f(honest)
    f = model.tikhonov_filter(wd.sv, lam)
    return {"k": k, "tsvd": e_k, "lam": lam, "tik": e_l, "best": e_f, "ratio": e_k / e_l, "width": int(np.sum((f > 0.1) & (f < 0.9)))}


def equal_experiment(n: int = C.DEFAULT_N) -> dict:
    """TSVD gegen Tikhonov gegen besten Filter: über Rauschstufen (ehrlich und selbes Gitter), über die Breite der Unschärfe und über drei Matrizen."""
    noise_rows = []
    for nz in C.EXP_NOISE_LEVELS:
        wd = World(n, C.DEFAULT_W, nz)
        noise_rows.append({"noise": nz, "picard": wd.picard(), "honest": _compare(wd, True), "crime": _compare(wd, False)})
    w_rows = []
    for w in C.EXP_W_LEVELS:
        wd = World(n, w, C.DEFAULT_NOISE)
        w_rows.append({"w": w, "cond": wd.cond, "honest": _compare(wd, True), "crime": _compare(wd, False)})
    op_rows = []
    for op in C.OPERATORS:
        for nz in C.EXP_NOISE_LEVELS:
            wd = _operator_world(op, n, nz)
            c = _compare(wd, False)
            op_rows.append({"op": op, "noise": nz, "cond": wd.cond, **c})
    ratios = [r["honest"]["ratio"] for r in noise_rows] + [r["crime"]["ratio"] for r in noise_rows] + [r["honest"]["ratio"] for r in w_rows] \
        + [r["crime"]["ratio"] for r in w_rows] + [r["ratio"] for r in op_rows]
    return {"noise_rows": noise_rows, "w_rows": w_rows, "op_rows": op_rows, "ratio_min": min(ratios), "ratio_max": max(ratios), "n": n}


# ----------------------------------------------------------------------------- Experiment 3: Rang statt Gitter

def grid_experiment(ns=C.EXP_N_SWEEP, w: float = C.DEFAULT_W, sigma: float | None = None) -> list[dict]:
    """Bester Rang und Fehler je Gitterfeinheit bei gleichem absolutem Rauschen (ehrliche Daten)."""
    sigma = standard_sigma() if sigma is None else sigma
    rows = []
    for n in ns:
        wd = World(n, w, sigma=sigma)
        c = _compare(wd, True)
        rows.append({"n": n, "cond": wd.cond, **c, "direct": wd.err(np.ones(n), True)})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: wie weit vom besten Filter, und die Gestalt

def filter_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    """Beste Filter gegen TSVD und Tikhonov (ehrlich und selbes Gitter, n = 48 und n = 24) und die mittlere Gestalt beider Lösungen über feste Ströme."""
    out = {"n": n, "reps": reps, "cases": []}
    for label, nn, honest in (("n = %d, ehrliche Daten" % n, n, True), ("n = %d, Daten aus demselben Gitter" % n, n, False), ("n = 24, ehrliche Daten", 24, True)):
        wd = World(nn, w, sigma=standard_sigma() if nn != n else None, noise=noise if nn == n else None)
        c = _compare(wd, honest)
        out["cases"].append({"label": label, "n": nn, "honest": honest, **c})
    wd = World(n, w, noise)
    k, _ = wd.best_k(True)
    lam, _ = wd.best_lam(True)
    b = draw_data(n, w, noise, SplitMix64(C.SEED_BASE + 4), crime=False, reps=reps)
    s = grid(n)
    i1, i2 = int(np.argmin(np.abs(s - 0.30))), int(np.argmin(np.abs(s - 0.58)))
    ib, il = int(np.argmin(np.abs(s - 0.78))), int(np.argmax(s >= 0.72))
    box = (s >= 0.72) & (s < 0.84)
    shapes = {}
    for name, f in (("tsvd", model.tsvd_filter(n, k)), ("tik", model.tikhonov_filter(wd.sv, lam))):
        sol = model.solve_filter(wd.u, wd.sv, wd.vt, b, f)
        mean = sol.mean(axis=0)
        rms = np.sqrt(np.mean((sol - wd.x) ** 2, axis=0))
        shapes[name] = {"mean": mean, "peak1": float(mean[i1]), "peak2": float(mean[i2]), "box": float(mean[ib]), "edge_before": float(mean[il - 1]),
                        "edge_at": float(mean[il]), "undershoot": float(mean.min()), "rms_box": float(np.mean(rms[box])),
                        "rms_rest": float(np.mean(rms[(s < 0.15) | (s > 0.9)])), "total": float(np.sqrt(np.mean(np.sum((sol - wd.x) ** 2, axis=1)) / np.sum(wd.x ** 2)))}
    out.update({"s": s, "truth": wd.x, "k": k, "lam": lam, "shapes": shapes, "peak1_truth": float(wd.x[i1]), "peak2_truth": float(wd.x[i2]), "box_truth": float(wd.x[ib])})
    return out


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    if a.k < a.k_star - 6 and a.bias > a.std:
        return "warning", (f"Zu wenige Komponenten: Der Bias ({err_pct(a.bias)}) dominiert die Streuung ({err_pct(a.std)}); der erwartete Fehler beträgt {err_pct(a.expected)}, "
                           f"beim besten Rang k* = {a.k_star} wären es {err_pct(a.err_star)}.")
    if a.k > a.k_star + 3 and a.std > a.bias:
        return "warning", (f"Zu viele Komponenten: Die Streuung ({err_pct(a.std)}) dominiert den Bias ({err_pct(a.bias)}); der erwartete Fehler beträgt {err_pct(a.expected)}, "
                           f"beim besten Rang k* = {a.k_star} wären es {err_pct(a.err_star)}.")
    if a.settings.data == "verbrechen" and a.err_honest_at_k > 1.3 * a.expected:
        return "info", (f"Mit Daten aus demselben Gitter beträgt der erwartete Fehler {err_pct(a.expected)}; mit ehrlichen Daten wären es {err_pct(a.err_honest_at_k)}. "
                        "Der Test mit demselben Gitter meldet eine Genauigkeit, die es nicht gibt.")
    return "success", (f"Der erwartete Fehler beträgt {err_pct(a.expected)} (Bias {err_pct(a.bias)}, Streuung {err_pct(a.std)}); die direkte Inversion hätte {err_pct(a.err_naive)}, "
                       f"Tikhonov beim besten Parameter {err_pct(a.err_tik_star)}. Bester Rang k* = {a.k_star}: {err_pct(a.err_star)}.")


def verdict_rank(res: dict) -> tuple[str, str]:
    by = {r["offset"]: r for r in res["rows"]}
    return "info", (f"Der beste Rang k* = {res['k_star']} gibt {err_pct(res['err_star'])}, Tikhonov beim besten Parameter {err_pct(res['err_tik'])}. "
                    f"Drei Komponenten zu viel kosten {de(100.0 * (by[3]['ratio'] - 1.0), 0)} %, fünf zu viel {de(100.0 * (by[5]['ratio'] - 1.0), 0)} %; "
                    f"fünf zu wenige {de(100.0 * (by[-5]['ratio'] - 1.0), 0)} %, zehn zu wenige {de(100.0 * (by[-10]['ratio'] - 1.0), 0)} %.")


def verdict_equal(res: dict) -> tuple[str, str]:
    return "success", (f"Über Rauschstufen, Breiten der Unschärfe und drei Matrizen liegt das Verhältnis des erwarteten Fehlers von abgeschnittener SVD zu Tikhonov "
                       f"zwischen {de(res['ratio_min'], 2)} und {de(res['ratio_max'], 2)}: kein Verfahren gewinnt systematisch.")


def verdict_grid(rows: list[dict]) -> tuple[str, str]:
    fine = [r for r in rows if r["n"] >= C.DEFAULT_N]
    ks = [r["k"] for r in fine]
    return "success", (f"Der beste Rang liegt für n = {fine[0]['n']} bis {fine[-1]['n']} bei {min(ks)} bis {max(ks)}, während die direkte Inversion bis auf "
                       f"{rel_text(rows[-1]['direct'])} wächst. Der Rang gehört zum Problem, nicht zum Gitter.")


def verdict_filter(res: dict) -> tuple[str, str]:
    h = res["cases"][0]
    c = res["cases"][1]
    return "info", (f"Mit Daten aus demselben Gitter liegen beide Verfahren {de(100.0 * (c['tsvd'] / c['best'] - 1.0), 0)} % "
                    f"über dem besten Filter ({err_pct(c['tsvd'])} gegen {err_pct(c['best'])}); mit ehrlichen Daten {de(100.0 * (h['tsvd'] / h['best'] - 1.0), 0)} % "
                    f"({err_pct(h['tsvd'])} gegen {err_pct(h['best'])}). Der Abstand ist, was ein Filter nur mit Wissen über Wahrheit und Modellfehler gewinnen könnte.")
