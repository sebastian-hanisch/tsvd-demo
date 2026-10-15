"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import tsv_constants as C
from tsv_evaluation import Analysis, de, lam_text, noise_label, sci

LOG_FLOOR = 1e-17


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Messwert b": a.b[:k],
        "Direkte Inversion": a.x_naive[:k],
        "Abgeschnittene SVD": a.x_tsvd[:k],
        "Tikhonov (λ*)": a.x_tik[:k],
    })


def style_table(df: pd.DataFrame):
    big = lambda v: sci(v, 1) if abs(v) >= 1e4 else de(v, 3)  # noqa: E731
    return df.style.format({"Ort s": lambda v: de(v, 3), "Wahrheit x": lambda v: de(v, 3), "Messwert b": lambda v: de(v, 4), "Direkte Inversion": big,
                            "Abgeschnittene SVD": lambda v: de(v, 3), "Tikhonov (λ*)": lambda v: de(v, 3)})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, abgeschnittene SVD, Tikhonov beim besten Parameter und (blass) die direkte Inversion; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    peak = float(np.max(np.abs(a.x_naive))) if np.all(np.isfinite(a.x_naive)) else float("inf")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x_naive, mode="lines", name="direkte Inversion", line=dict(color=C.COLOR_NAIVE, width=1, dash="dot"), opacity=0.5, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_tik, mode="lines", name=f"Tikhonov (λ* = {lam_text(a.lam_star)})", line=dict(color=C.COLOR_TIK, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_tsvd, mode="lines+markers", name=f"abgeschnittene SVD (k = {a.k})", line=dict(color=C.COLOR_TSVD, width=2.5), marker=dict(size=5),
                             hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak_text = "unendlich" if not np.isfinite(peak) else (de(peak, 1) if peak < 1000.0 else sci(peak))
    note = f"Die direkte Inversion reicht bis ±{peak_text} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten." if peak > max(abs(lo), abs(hi)) else ""
    return _fixed(fig, 380, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def build_filter_chart(a: Analysis) -> go.Figure:
    """Filterfaktoren je Komponente: der Schnitt (Stufe), die glatte Strafe, der beste Filter (Orakel); die Marke k* der Picard-Zahl."""
    idx = np.arange(1, a.settings.n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=a.f_tsvd, mode="lines", line_shape="hv", name=f"abgeschnittene SVD (k = {a.k})", line=dict(color=C.COLOR_TSVD, width=3)))
    fig.add_trace(go.Scatter(x=idx, y=a.f_tik, mode="lines+markers", name=f"Tikhonov (λ* = {lam_text(a.lam_star)})", line=dict(color=C.COLOR_TIK, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=idx, y=a.f_best, mode="lines", name="bester Filter (Orakel)", line=dict(color=C.COLOR_BEST, width=2, dash="dot")))
    fig.add_vline(x=a.picard + 0.5, line=dict(color="#888888", dash="dot"), annotation_text=f"k* aus Stück 1 = {a.picard}", annotation_position="top right")
    fig.update_xaxes(title="Index i der Komponente")
    fig.update_yaxes(title="Filterfaktor fᵢ", range=[-0.1, 1.15])
    return _fixed(fig, 360)


def build_rank_chart(a: Analysis, reveal: bool) -> go.Figure:
    """Erwarteter Fehler, Bias und Streuung über den Rang k (logarithmisch); der gewählte Rang, bei „Orakel zeigen“ auch das Optimum."""
    c = a.curve
    k = c["k"][1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=np.maximum(c["total"][1:], LOG_FLOOR), mode="lines+markers", name="erwarteter Fehler", line=dict(color="#222222", width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=k, y=np.maximum(c["bias"][1:], LOG_FLOOR), mode="lines", name="Bias", line=dict(color=C.COLOR_BEST, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=k, y=np.maximum(c["std"][1:], LOG_FLOOR), mode="lines", name="Streuung", line=dict(color=C.COLOR_NAIVE, width=2, dash="dot")))
    if a.k >= 1:
        fig.add_vline(x=a.k, line=dict(color=C.COLOR_TSVD, width=2), annotation_text=f"k = {a.k}", annotation_position="top left")
    if reveal:
        fig.add_vline(x=a.k_star, line=dict(color=C.COLOR_TRUTH, width=2, dash="dash"), annotation_text=f"k* = {a.k_star}", annotation_position="top right")
    fig.update_xaxes(title="Rang k (Zahl der behaltenen Komponenten)")
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", range=[-2.5, 1.5], tickvals=[0.01, 0.1, 1, 10], ticktext=["1 %", "10 %", "100 %", "1000 %"])
    return _fixed(fig, 380)


def build_offset_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["offset"] for r in rows], y=[100.0 * r["closed"] for r in rows], mode="lines+markers", name="abgeschnittene SVD, geschlossen",
                             line=dict(color=C.COLOR_TSVD, width=2.5)))
    fig.add_trace(go.Scatter(x=[r["offset"] for r in rows], y=[100.0 * r["sim"] for r in rows], mode="markers", name=f"simuliert ({res['reps']} Wdh.)",
                             marker=dict(size=9, color="#222222", symbol="diamond")))
    fig.add_hline(y=100.0 * res["err_tik"], line=dict(color=C.COLOR_TIK, dash="dash"), annotation_text="Tikhonov beim besten λ", annotation_position="top left")
    fig.update_xaxes(title="Abstand vom besten Rang k*", tickvals=[r["offset"] for r in rows])
    fig.update_yaxes(title="Erwarteter Fehler (% der Signalnorm)", range=[0, max(100.0 * max(r["closed"] for r in rows) * 1.1, 14.0)])
    return _fixed(fig, 360)


def build_equal_chart(res: dict) -> go.Figure:
    """Verhältnis des erwarteten Fehlers von abgeschnittener SVD zu Tikhonov (beide beim besten Parameter) über die Rauschstufe."""
    fig = go.Figure()
    xs = [100.0 * r["noise"] for r in res["noise_rows"]]
    fig.add_trace(go.Scatter(x=xs, y=[r["honest"]["ratio"] for r in res["noise_rows"]], mode="lines+markers", name="Gauß, ehrliche Daten", line=dict(color=C.COLOR_TIK, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[r["crime"]["ratio"] for r in res["noise_rows"]], mode="lines+markers", name="Gauß, Daten aus demselben Gitter", line=dict(color=C.COLOR_TSVD, width=2.5)))
    for op, color in (("integral", "#2F6B65"), ("integral2", C.COLOR_BEST)):
        rows = [r for r in res["op_rows"] if r["op"] == op]
        fig.add_trace(go.Scatter(x=[100.0 * r["noise"] for r in rows], y=[r["ratio"] for r in rows], mode="lines+markers", name=C.OPERATOR_LABELS[op],
                                 line=dict(color=color, width=2, dash="dot")))
    fig.add_hline(y=1.0, line=dict(color="#888888", dash="dash"), annotation_text="gleich gut", annotation_position="bottom right")
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="log", tickvals=xs, ticktext=[noise_label(r["noise"]).replace(" %", "") for r in res["noise_rows"]])
    fig.update_yaxes(title="Fehler TSVD / Fehler Tikhonov", range=[0.8, 1.3])
    return _fixed(fig, 380)


def build_grid_chart(rows: list[dict]) -> go.Figure:
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(n) for n in ns], y=[r["k"] for r in rows], marker_color=C.COLOR_TSVD, text=[str(r["k"]) for r in rows], textposition="outside",
                         name="bester Rang k*"))
    fig.update_xaxes(title="Gitterpunkte n", type="category")
    fig.update_yaxes(title="Bester Rang (n bedeutet: kein Schnitt nötig)", range=[0, max(ns) * 0.5])
    return _fixed(fig, 340, showlegend=False)


def build_shape_chart(res: dict) -> go.Figure:
    s = res["s"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=res["truth"], mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s, y=res["shapes"]["tik"]["mean"], mode="lines+markers", name=f"Mittel Tikhonov (λ* = {lam_text(res['lam'])})",
                             line=dict(color=C.COLOR_TIK, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=s, y=res["shapes"]["tsvd"]["mean"], mode="lines+markers", name=f"Mittel abgeschnittene SVD (k = {res['k']})",
                             line=dict(color=C.COLOR_TSVD, width=2, dash="dash"), marker=dict(size=4, symbol="diamond")))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert")
    return _fixed(fig, 380)
