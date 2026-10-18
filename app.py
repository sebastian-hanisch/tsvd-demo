"""Abgeschnittene SVD: der harte Schnitt gegen das Rauschen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 und 2 (Phantom, Gauß-Unschärfe, Rauschen) wird die direkte Inversion nach
dem k-ten Singulärwert abgeschnitten und mit Tikhonov und dem besten Filter verglichen. Kern: tsv_model.py, Generator: inverse_generator.py (wortgleich in allen
Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import tsv_constants as C
from tsv_evaluation import (
    Settings,
    analyse,
    de,
    equal_experiment,
    err_pct,
    filter_experiment,
    grid_experiment,
    lam_text,
    noise_label,
    rank_experiment,
    rel_text,
    sci,
    standard_sigma,
    verdict_equal,
    verdict_filter,
    verdict_grid,
    verdict_instance,
    verdict_rank,
)
from tsv_pdf_export import build_pdf
from tsv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from tsv_visualization import (
    build_equal_chart,
    build_filter_chart,
    build_grid_chart,
    build_offset_chart,
    build_rank_chart,
    build_recon_chart,
    build_shape_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Abgeschnittene SVD – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _rank():
    return rank_experiment()


@st.cache_data(show_spinner=SPINNER)
def _equal():
    return equal_experiment()


@st.cache_data(show_spinner=SPINNER)
def _grid():
    return grid_experiment()


@st.cache_data(show_spinner=SPINNER)
def _filter():
    return filter_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("✂️ Abgeschnittene SVD: der harte Schnitt gegen das Rauschen")
st.markdown(
    """
**Der harte Schnitt und die glatte Strafe sind dasselbe Verfahren in zwei Gewändern.** Dies ist das dritte Stück der Linie **Inverse Probleme und
Regularisierung** und baut auf **Das schlecht gestellte Problem** und **Tikhonov-Regularisierung** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen.
Die **abgeschnittene SVD** behält die ersten k Komponenten der direkten Inversion und wirft den Rest weg; Tikhonov dämpft sie stufenlos. Die Welt ist simuliert; das
Phantom ist die bekannte Wahrheit.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die abgeschnittene SVD", expanded=True):
    st.markdown(
        r"""
1. **Zerlegen.** Die Matrix wird in Singulärwerte $\sigma_i$ und Singulärvektoren $u_i, v_i$ zerlegt; die direkte Inversion ist $\sum_i (u_i^\top b/\sigma_i)\,v_i$.
2. **Schneiden.** Die abgeschnittene SVD behält nur die ersten $k$ Summanden: der Filterfaktor ist $f_i = 1$ für $i \le k$ und sonst $0$. Das Rauschen der kleinen
   Singulärwerte kommt nicht mehr durch, dafür fehlen die feinen Strukturen (Bias).
3. **Filter lesen.** Tikhonov ist derselbe Ansatz mit einem stufenlosen Filter $f_i = \sigma_i^2/(\sigma_i^2 + \lambda^2)$. Das Diagramm „Filter“ legt beide übereinander.
4. **Der Rang ist nicht gegeben.** Der beste Rang $k^*$ braucht die Wahrheit; wie man ihn ohne sie findet, zeigt das Folgestück **Parameterwahl**.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Wie viele Messpunkte und Unbekannte. Der beste Rang hängt kaum vom Gitter ab.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Daten**")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrliche Daten kommen aus der analytischen Unschärfe und sind vom Gitter unabhängig. Daten aus demselben Gitter entstehen aus "
                             "der Matrix, die danach invertiert wird (inverse crime).")
    st.markdown("**Schnitt**")
    k = st.slider("Rang k (behaltene Komponenten)", *bounds("k_slider"), step=C.K_STEP, key="k_slider",
                  help="Wie viele Singulärkomponenten die Lösung behält. 0 gibt die Nulllösung; ein Wert von n oder mehr gibt die direkte Inversion von Stück 1.")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data, "k_slider": int(k), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, int(k), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fehler der abgeschnittenen SVD", err_pct(a.err_tsvd), delta=f"direkte Inversion: {err_pct(a.err_naive)}", delta_color="off",
          help="‖x̂ − x‖ / ‖x‖ dieser Stichprobe, mit der Wahrheit x des Phantoms.")
c2.metric("Erwarteter Fehler", err_pct(a.expected), delta=f"Bias {err_pct(a.bias)}, Streuung {err_pct(a.std)}", delta_color="off",
          help="Geschlossene Formel: Wurzel aus Bias² plus Varianz, relativ zur Signalnorm, gemittelt über das Rauschen.")
c3.metric("Freiheitsgrade", f"{a.k} von {settings.n}", delta=f"Tikhonov beim besten λ: {de(float(a.f_tik.sum()), 1)}", delta_color="off",
          help="Summe der Filterfaktoren: bei der abgeschnittenen SVD genau k, bei Tikhonov stufenlos.")
c4.metric("Vergleich: Tikhonov, bester Parameter", err_pct(a.err_tik_star), delta=f"λ* = {lam_text(a.lam_star)}", delta_color="off",
          help="Erwarteter Fehler von Tikhonov beim besten λ (Orakel), derselben Datenart.")

tab_rec, tab_fil, tab_rank, tab_tab = st.tabs(["Rekonstruktion", "Filter", "Fehler über k", "Tabelle"])
reveal = st.session_state.get("reveal_oracle", False)
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, orange die abgeschnittene SVD, blau gestrichelt Tikhonov beim besten Parameter, blass rot die direkte Inversion. Die Achse ist fest.")
with tab_fil:
    st.plotly_chart(build_filter_chart(a), width="stretch", key="filter_chart")
    st.caption("Filterfaktoren der Komponenten: der Schnitt (Stufe), die glatte Strafe und der beste Filter, der die Wahrheit kennt. Die graue Marke ist die Picard-Zahl aus Stück 1.")
with tab_rank:
    st.plotly_chart(build_rank_chart(a, reveal), width="stretch", key="rank_chart")
    st.caption("Erwarteter Fehler (schwarz), Bias (violett) und Streuung (rot) über den Rang, geschlossen gerechnet. Links dominiert der Bias, rechts die Streuung.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz; „Direkte Inversion“ in Zehnerpotenzen, wenn der Betrag über 10 000 liegt.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="abgeschnittene-svd.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist der Rang gut gewählt?")
st.toggle("🙈 Orakel zeigen (bester Rang aus der Wahrheit)", key="reveal_oracle")
if st.session_state.get("reveal_oracle", False):
    r1, r2, r3 = st.columns(3)
    r1.metric("Bester Rang k*", str(a.k_star), help="Minimum des erwarteten Fehlers über k; er braucht die Wahrheit und existiert in der Praxis nicht.")
    r2.metric("Erwarteter Fehler bei k*", err_pct(a.err_star))
    r3.metric("Abstand: k minus k*", f"{a.k - a.k_star:+d}".replace("-", "−"))
r4, r5 = st.columns(2)
r4.metric("Erwarteter Fehler mit ehrlichen Daten", err_pct(a.err_honest_at_k), help="Zum gewählten Rang, mit der analytischen Unschärfe als Daten.")
r5.metric("Erwarteter Fehler mit Daten aus demselben Gitter", err_pct(a.err_crime_at_k), help="Zum gewählten Rang, mit Daten aus der Matrix, die invertiert wird.")
verdict(verdict_instance(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der Schnitt und seine Nachsicht")
res1 = _rank()
st.caption(f"Standardinstanz (n = {res1['n']}), ehrliche Daten, {res1['reps']} feste Wiederholungen je Rang. Gezeigt ist der Fehler bei Rängen um den besten Rang "
           f"k* = {res1['k_star']}; die gestrichelte Linie ist Tikhonov beim besten λ.")
st.plotly_chart(build_offset_chart(res1), width="stretch", key="offset_chart")
verdict(verdict_rank(res1))
st.dataframe(
    [{"Abstand von k*": f"{r['offset']:+d}".replace("-", "−"), "Rang k": r["k"], "Erwartet": err_pct(r["closed"]), "Simuliert": err_pct(r["sim"]),
      "Verhältnis zum Optimum": de(r["ratio"], 2)} for r in res1["rows"]],
    hide_index=True, width="stretch")
st.caption("Zum Vergleich Tikhonov: Mit dem 0,3-fachen des besten λ liegt der Fehler "
           f"{de(100.0 * (next(r['ratio'] for r in res1['tik_rows'] if r['factor'] == 0.3) - 1.0), 0)} % über dem Optimum, mit dem 10-fachen "
           f"{de(100.0 * (next(r['ratio'] for r in res1['tik_rows'] if r['factor'] == 10.0) - 1.0), 0)} %. Zu viele Komponenten sind teurer als zu wenige, aber erst "
           "ab etwa drei.")

st.subheader("🔬 Experiment 2: Zwei Filter, ein Ergebnis")
res2 = _equal()
st.caption(f"Erwarteter Fehler der abgeschnittenen SVD geteilt durch den von Tikhonov, je beim besten Parameter (n = {res2['n']}): über Rauschstufen mit Gauß-Unschärfe "
           "(ehrliche Daten und Daten aus demselben Gitter) und mit zwei anderen Matrizen, deren Singulärwerte nur algebraisch fallen (gleiches Phantom).")
st.plotly_chart(build_equal_chart(res2), width="stretch", key="equal_chart")
verdict(verdict_equal(res2))
rows2 = []
for r in res2["noise_rows"]:
    rows2.append({"Matrix": "Gauß (ehrlich)", "Rauschen": noise_label(r["noise"]), "k*": r["honest"]["k"], "TSVD": err_pct(r["honest"]["tsvd"]), "Tikhonov": err_pct(r["honest"]["tik"]),
                  "Bester Filter": err_pct(r["honest"]["best"]), "Verhältnis": de(r["honest"]["ratio"], 2), "Übergang Tikhonov": f"{r['honest']['width']} Komp."})
for r in res2["op_rows"]:
    rows2.append({"Matrix": C.OPERATOR_LABELS[r["op"]], "Rauschen": noise_label(r["noise"]), "k*": r["k"], "TSVD": err_pct(r["tsvd"]), "Tikhonov": err_pct(r["tik"]),
                  "Bester Filter": err_pct(r["best"]), "Verhältnis": de(r["ratio"], 2), "Übergang Tikhonov": f"{r['width']} Komp."})
st.dataframe(rows2, hide_index=True, width="stretch", height=420)
st.caption("„Übergang Tikhonov“ zählt die Komponenten, deren Filterfaktor zwischen 0,1 und 0,9 liegt: bei langsam fallenden Singulärwerten ist der glatte Filter breit, und "
           "der Schnitt ist trotzdem nicht schlechter. Wo k* = n ist, gibt es nichts zu schneiden und beide Verfahren sind gleich.")

st.subheader("🔬 Experiment 3: Rang statt Gitter")
rows3 = _grid()
st.caption(f"Standardunschärfe (w = {de(C.DEFAULT_W, 3)}), absolutes Rauschen der Standardinstanz ({sci(standard_sigma())}), ehrliche Daten, bester Rang und erwarteter Fehler je Gitter.")
st.plotly_chart(build_grid_chart(rows3), width="stretch", key="grid_chart")
verdict(verdict_grid(rows3))
st.dataframe(
    [{"n": r["n"], "Kondition": sci(r["cond"]) if r["cond"] >= 1000 else de(r["cond"], 1), "Bester Rang k*": r["k"], "Abgeschnittene SVD": err_pct(r["tsvd"]),
      "Tikhonov": err_pct(r["tik"]), "Bester Filter": err_pct(r["best"]), "Direkte Inversion": rel_text(r["direct"])} for r in rows3],
    hide_index=True, width="stretch")
st.caption("Bei n = 24 und n = 32 braucht es kaum einen Schnitt: das grobe Gitter regularisiert selbst. Dass n = 32 hier besser abschneidet als n = 48 bis 96, hängt an der "
           "Lage der Kastenkanten zum Gitter und ist kein allgemeiner Befund.")

st.subheader("🔬 Experiment 4: Wie weit vom besten Filter")
res4 = _filter()
st.caption(f"Erwarteter Fehler beider Verfahren gegen den besten Diagonalfilter, der Wahrheit und Modellfehler kennt (Untergrenze für dämpfende Filter, kein Verfahren); darunter die mittlere "
           f"Gestalt der Lösungen über {res4['reps']} feste Wiederholungen mit ehrlichen Daten.")
verdict(verdict_filter(res4))
st.dataframe(
    [{"Fall": c["label"], "Bester Rang": c["k"], "Abgeschnittene SVD": err_pct(c["tsvd"]), "Tikhonov": err_pct(c["tik"]), "Bester Filter": err_pct(c["best"]),
      "Abstand zum besten Filter": f"{de(100.0 * (c['tsvd'] / c['best'] - 1.0), 0)} %"} for c in res4["cases"]],
    hide_index=True, width="stretch")
st.plotly_chart(build_shape_chart(res4), width="stretch", key="shape_chart")
sh = res4["shapes"]
st.dataframe(
    [{"Größe": "Hügel 1 (Wahrheit {})".format(de(res4["peak1_truth"], 3)), "Abgeschnittene SVD": de(sh["tsvd"]["peak1"], 3), "Tikhonov": de(sh["tik"]["peak1"], 3)},
     {"Größe": "Hügel 2, schmal (Wahrheit {})".format(de(res4["peak2_truth"], 3)), "Abgeschnittene SVD": de(sh["tsvd"]["peak2"], 3), "Tikhonov": de(sh["tik"]["peak2"], 3)},
     {"Größe": "Kasten, Mitte (Wahrheit {})".format(de(res4["box_truth"], 2)), "Abgeschnittene SVD": de(sh["tsvd"]["box"], 3), "Tikhonov": de(sh["tik"]["box"], 3)},
     {"Größe": "Unterschwinger (Minimum des Mittels)", "Abgeschnittene SVD": de(sh["tsvd"]["undershoot"], 3), "Tikhonov": de(sh["tik"]["undershoot"], 3)},
     {"Größe": "Punktfehler am Kasten (RMS)", "Abgeschnittene SVD": de(sh["tsvd"]["rms_box"], 4), "Tikhonov": de(sh["tik"]["rms_box"], 4)},
     {"Größe": "Punktfehler am Rand (RMS)", "Abgeschnittene SVD": de(sh["tsvd"]["rms_rest"], 4), "Tikhonov": de(sh["tik"]["rms_rest"], 4)}],
    hide_index=True, width="stretch")
st.caption("Der Schnitt und die glatte Strafe sehen sich auf dem Bild zum Verwechseln ähnlich; der Kasten wird beim Schnitt etwas schlechter getroffen. Was beide nicht können, "
           "kann nur ein Filter, der mehr über die Wahrheit weiß: eine Strafe, die Sprünge schont, ist der Inhalt des Folgestücks **Sparsity und Total Variation**.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Rang ist bekannt** | Der beste Rang braucht die Wahrheit. Ohne sie muss man ihn aus den Daten schätzen; zehn Komponenten zu wenig kosten {de(100.0 * (res1['rows'][0]['ratio'] - 1.0), 0)} %. | Parameterwahl |
| **Gleichwertig gilt für diese Matrizen** | Drei Matrizen und ein Phantom; andere Spektren oder Phantome können den Abstand verschieben (bis {de(res2['ratio_max'], 2)} in Experiment 2). | – |
| **Beide Verfahren sind Filter** | Iterationsverfahren mit frühem Abbruch sind ebenfalls Filter, aber mit einer eigenen Form. | Iterative Regularisierung |
| **Kanten und Spitzen leiden in beiden** | Der Kasten verliert in der Mitte etwa {de(100.0 * (1.0 - sh['tsvd']['box'] / res4['box_truth']), 0)} % (Schnitt) gegen {de(100.0 * (1.0 - sh['tik']['box'] / res4['box_truth']), 0)} % (Tikhonov). | Sparsity und Total Variation |
| **Der beste Filter ist ein Orakel** | Er kennt Wahrheit und Modellfehler; sein Abstand ist eine Untergrenze, kein erreichbares Ziel. | – |
| **Eine Dimension, ein Phantom** | Zwei Hügel und ein Kasten, weißes Rauschen; andere Formen verschieben die Zahlen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gegeben $A = U\,\mathrm{diag}(\sigma_1 \ge \dots \ge \sigma_n)\,V^\top$ und Daten $b = A x + r + e$ mit Modellfehler $r$ (null bei Daten aus demselben Gitter)
und Rauschen $e \sim N(0, \sigma_e^2 I)$. Mit $c_i = v_i^\top x$ und $\rho_i = u_i^\top r/\sigma_i$.

**Filterlösung.** $x_f = \sum_i f_i\,\frac{u_i^\top b}{\sigma_i}\,v_i$. Abgeschnittene SVD: $f_i = 1$ für $i \le k$, sonst $0$, also $x_k = V_k \Sigma_k^{-1} U_k^\top b$ (die
Pseudoinverse der besten Rang-$k$-Näherung von $A$). Tikhonov: $f_i = \sigma_i^2/(\sigma_i^2 + \lambda^2)$.

**Bias und Varianz.** Für beliebige Filterfaktoren ist $\text{Bias}_i = f_i(c_i + \rho_i) - c_i$ und $\text{Var}_i = f_i^2\sigma_e^2/\sigma_i^2$; der erwartete Fehler
$\sqrt{\sum_i \text{Bias}_i^2 + \text{Var}_i}\,/\,\|x\|$. Bei der abgeschnittenen SVD ist $\text{Bias}_i = \rho_i$ für $i \le k$ und $-c_i$ für $i > k$, die Varianz
$\sigma_e^2\sum_{i \le k} \sigma_i^{-2}$.

**Bester Filter.** Je Komponente minimiert $(f(c_i + \rho_i) - c_i)^2 + f^2\sigma_e^2/\sigma_i^2$ den Fehler bei
$\frac{c_i(c_i + \rho_i)}{(c_i + \rho_i)^2 + \sigma_e^2/\sigma_i^2}$; auf $[0, 1]$ begrenzt (nur Dämpfung, keine Verstärkung) ist das der beste Filter $f_i^*$.
Er braucht $x$ und $r$ und ist die Untergrenze aller dämpfenden Diagonalfilter, auch von Tikhonov und der abgeschnittenen SVD.

**Freiheitsgrade.** $\mathrm{df} = \sum_i f_i$; bei der abgeschnittenen SVD genau $k$.

**Literatur.** Das Beispiel ist erfunden; die abgeschnittene SVD und ihre Filterform stehen bei P. C. Hansen, *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM 1998,
und *Discrete Inverse Problems: Insight and Algorithms*, SIAM 2010; die Rang-$k$-Approximation bei C. Eckart und G. Young, Psychometrika 1, 1936
(alles aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html)."
)
