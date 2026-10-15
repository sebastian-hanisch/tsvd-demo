"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from tsv_evaluation import Analysis, de, err_pct, lam_text, rel_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0", "¹": "1", "²": "2",
            "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Abgeschnittene SVD"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    kind = "Daten aus demselben Gitter (inverse crime)" if s.data == "verbrechen" else "ehrliche Daten (analytisch)"
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, {kind}, "
                            f"Rang k = {a.k}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Fehler der abgeschnittenen SVD dieser Stichprobe: {rel_text(a.err_tsvd)}; Tikhonov beim besten Parameter: {rel_text(a.err_tik)}; direkte Inversion: {rel_text(a.err_naive)}.",
        f"Erwarteter Fehler (geschlossen): {err_pct(a.expected)} = Bias {err_pct(a.bias)} und Streuung {err_pct(a.std)}; Freiheitsgrade {de(a.df, 0)}.",
        f"Bester Rang (Wahrheit bekannt): k* = {a.k_star} mit erwartetem Fehler {err_pct(a.err_star)}; Tikhonov lambda* = {lam_text(a.lam_star)} mit {err_pct(a.err_tik_star)}; "
        f"bester Filter {err_pct(a.err_best)}.",
        f"Komponenten über dem Rauschen (Picard-Zahl aus Stück 1): {a.picard} von {s.n}.",
        f"Kondition der Matrix: {sci(a.cond)}; absolutes Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Messwert b", "Direkt", "TSVD", "Tikhonov"], [14, 22, 26, 28, 34, 24, 24]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        nv = a.x_naive[i]
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.b[i], 4), sci(nv, 1) if abs(nv) >= 1e4 else de(nv, 3), de(a.x_tsvd[i], 3), de(a.x_tik[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Abgeschnittene SVD: x = V_k diag(1/sigma) U_k^T b. Quelle: tsvd-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
