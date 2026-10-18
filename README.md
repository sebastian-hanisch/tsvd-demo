# Abgeschnittene SVD: der harte Schnitt gegen das Rauschen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-tsvd-demo.streamlit.app/)**

Drittes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo) und [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist.

## Kernfrage

Die direkte Inversion verstärkt das Rauschen der kleinen Singulärwerte. Ein harter Schnitt nach der k-ten Komponente entfernt es; Tikhonov dämpft es stufenlos. Wie gut ist der Schnitt, wie genau muss k stimmen, ist er schlechter als die glatte Strafe, hängt der Rang vom Gitter ab, und wie weit liegt beides vom besten möglichen Filter?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen, zwei Datenarten):

- **Wahrheit** x(s) auf [0, 1]: zwei Gauß-Hügel und ein Kasten; **Messung** g(t) = ∫ k(t − s) x(s) ds mit dem Gauß-Kern der Breite w, analytisch bekannt.
- **Gitter** mit n Punkten: A = h K. Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴.
- **Daten:** *ehrlich* (analytisch, vom Gitter unabhängig) oder *aus demselben Gitter* (b = A x + e, „inverse crime“); das Rauschen ist in beiden gleich.
- **Filterlösung** x_f = V diag(f/σ) Uᵀ b. Abgeschnittene SVD: fᵢ = 1 für i ≤ k, sonst 0. Tikhonov: fᵢ = σᵢ²/(σᵢ² + λ²).

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Geschlossene Bias-Varianz-Zerlegung für beliebige Filter:** Biasᵢ = fᵢ(cᵢ + ρᵢ) − cᵢ, Varᵢ = fᵢ²σ²/σᵢ² mit c = Vᵀx und ρᵢ = (Uᵀr)ᵢ/σᵢ (Modellfehler r der ehrlichen Daten). Die App zeigt Erwartungswerte; die Stichprobe der Primäransicht ist ein einzelner Strom.
- **Bester Rang k\*, bester Parameter λ\*:** Minimum des erwarteten Fehlers (Aufzählung, Gitter plus goldener Schnitt); beide brauchen die Wahrheit und sind Orakel.
- **Bester Filter:** je Komponente das Minimum der Parabel, auf [0, 1] begrenzt (nur Dämpfung); er kennt Wahrheit und Modellfehler und ist die Untergrenze aller dämpfenden Diagonalfilter.
- **Zwei zusätzliche Matrizen** nur in Experiment 2 (gleiches Phantom, Daten b = A x + e): Integration (σᵢ ~ 1/i) und doppelte Integration (σᵢ ~ 1/i²).
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 400 Wiederholungen je Punkt, nie ein einzelner Seed.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, ehrliche Daten), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wie gut ist der beste Schnitt? | Bester Rang **k\* = 33** von 48, erwarteter Fehler **11,4 %** (Bias 11,3 %, Streuung 1,9 %); Tikhonov beim besten λ\* = 0,0095: 11,7 %; direkte Inversion 247 % (Stichprobe mit Seed 35: 258 %, Schnitt 11,5 %, Tikhonov 11,8 %). Freiheitsgrade 33 gegen 32,9. Die Picard-Zahl aus Stück 1 ist 31. | `test_standard_instance_expected_error_bias_std_and_dof`, `test_best_rank_parameter_and_best_filter_on_the_standard_instance`, `test_experiment_1_best_rank_and_endpoints` |
| Wie genau muss k stimmen (Experiment 1)? | k\* − 10: +32 %; −5: +10 %; −3: +4 %; −1: +2 %; +1: 0 %; +2: +2 %; **+3: +6 %**; **+5: +24 %**. Tikhonov: λ\* mal 0,1 +73 %, mal 0,3 +10 %, mal 10 +16 %. Zu viele Komponenten sind teurer als zu wenige, aber erst ab etwa drei. | `test_experiment_1_cost_of_missing_the_rank`, `test_experiment_1_tikhonov_comparison` |
| Stimmt die geschlossene Formel? | Simulation (400 Wdh.) und Formel liegen bei allen Rängen von k\* − 10 bis k\* + 5 auf 1 % beieinander; k = 0 gibt die Nulllösung (Fehler 1,000), k = n die direkte Inversion (2,47). | `test_experiment_1_simulation_confirms_the_closed_form`, `test_experiment_1_best_rank_and_endpoints` |
| Ist der Schnitt schlechter als Tikhonov (Experiment 2)? | Über Rauschstufen, Breiten der Unschärfe und drei Matrizen liegt das Verhältnis der erwarteten Fehler **zwischen 0,91 und 1,19**. Gauß, ehrliche Daten, Rauschen 10 % bis 0,001 %: 0,93 / 1,04 / 0,98 / 0,98 / 1,00; Daten aus demselben Gitter 0,92 / 1,06 / 1,00 / 1,02 / 1,11. | `test_experiment_2_ratio_range`, `test_experiment_2_ratios_by_noise_with_gauss` |
| Auch bei anderen Matrizen? | Integration: 0,91 / 0,95 / 1,02 / 1,00 / 1,00; doppelte Integration: 1,04 / 0,92 / 0,94 / **1,19** / 1,00. Der glatte Filter ist dort bis 40 Komponenten breit, der Schnitt trotzdem nicht schlechter. Wo k\* = n ist, gibt es nichts zu schneiden. | `test_experiment_2_other_matrices` |
| Und über die Unschärfe? | w = 0,025 / 0,03 / 0,035 / 0,04: 0,98 / 0,98 / 1,00 / 1,01 (ehrlich), 1,04 / 1,00 / 1,065 / 1,02 (selbes Gitter); bester Rang 34 / 33 / 29 / 28. | `test_experiment_2_ratios_by_width` |
| Ist der Rang die Picard-Zahl? | Bester Rang 15 / 28 / 33 / 34 / 47 gegen Picard-Zahl 16 / 26 / 31 / 40 / 46: ±2 bis Rauschen 0,1 %, bei 0,01 % um 6 daneben, weil der Modellfehler den Boden setzt. | `test_experiment_2_best_ranks_and_picard_numbers` |
| Hängt der Rang vom Gitter ab (Experiment 3)? | Nein: bester Rang 24 (kein Schnitt) / 30 / **33 / 30 / 33 / 30** für n = 24 / 32 / 48 / 64 / 80 / 96. Fehler Schnitt: 9,4 / 4,1 / 11,4 / 9,6 / 10,2 / 10,3 %; die direkte Inversion wächst bis über 10¹². | `test_experiment_3_best_rank_by_grid`, `test_experiment_3_errors_by_grid` |
| Wie weit vom besten Filter (Experiment 4)? | Bester dämpfender Filter: **9,7 %** (ehrlich) und 6,9 % (selbes Gitter). Schnitt und Tikhonov liegen mit Daten aus demselben Gitter 2 % darüber, mit ehrlichen Daten **18 %** (n = 24: 16 %). | `test_experiment_4_best_filter_gap` |
| Gestalt der Lösung? | Mittel über 400 Wdh.: Hügel 1 1,001 / 0,999 (Schnitt / Tikhonov), Hügel 2 0,674 / 0,676, Kasten Mitte 0,453 / 0,462 (0,5), Unterschwinger −0,026 / −0,029; Punktfehler am Kasten 0,0295 / 0,0258 (**14 % größer** beim Schnitt), am Rand 0,008. | `test_experiment_4_shapes` |
| Presets | k = 15: erwartet 16,0 %, ganz Bias (Streuung unter 0,2 %); k = 40: 19,0 % (Streuung 14,8 %, Bias 11,9 %); n = 96, k = 33: 10,4 %, die direkte Inversion hätte 10¹². | `test_preset_numbers_in_the_help_texts` |
| Mini-Instanz von Hand | 3 × 3-Matrix (Singulärwerte 1 / 0,5 / 0,1): k = 2 gibt (1,333, 1,333, 1,333), Bias 33,3 % der Norm, Varianz 5 σ²; k = 1 dieselbe Lösung mit Varianz σ²; k = 3 die direkte Inversion mit 105 σ². | `test_mini_instance_by_hand_numbers`, `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **„Gleichwertig“ gilt fast immer, nicht unter Bedingungen.** Der Plan fragte, wann der Schnitt gleichwertig ist; die Messung sagt: in allen 33 Zellen zwischen 0,91 und 1,19.
- **Der erste „beste Filter“ war zu gut.** Ohne Begrenzung auf [0, 1] verstärkte er Komponenten (f bis 7), weil er den Modellfehler kennt, und lag 40 % unter dem Schnitt. Auf Dämpfung begrenzt sind es 18 % (9,7 % statt 8,1 %); der Plan nannte noch die ungenaue Zahl.
- **Orakelprüfung gegen `pinv`:** Die erste Fassung scheiterte mit Abweichungen bis 10¹², weil `rcond` zwischen σₖ und dem numerischen Nullbereich liegen muss.
- **n = 32 ist kein Gegenbeispiel:** der Fehler 4,1 % bei n = 32 hängt an der Lage der Kastenkanten zum Gitter (Stück 2).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Rang ist bekannt** | Der beste Rang braucht die Wahrheit. | Parameterwahl |
| **„Gleichwertig“ gilt für diese Matrizen** | Drei Matrizen und ein Phantom; andere Spektren oder Phantome können den Abstand verschieben (bis 1,19 hier). | – |
| **Beide Verfahren sind Filter** | Iterationsverfahren mit frühem Abbruch sind es auch, aber mit eigener Form. | Iterative Regularisierung |
| **Kanten und Spitzen leiden in beiden** | Der Kasten verliert in der Mitte rund 9 % (Schnitt) gegen 8 % (Tikhonov). | Sparsity und Total Variation |
| **Der beste Filter ist ein Orakel** | Er kennt Wahrheit und Modellfehler; sein Abstand ist eine Untergrenze. | – |
| **Eine Dimension, ein Phantom** | Zwei Hügel und ein Kasten, weißes Rauschen. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo): Stück 1; liefert die Welt und die Picard-Zahl k\*; hier nur ein Vergleichswert.
- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo): Stück 2; die glatte Strafe, deren Filter hier neben dem Schnitt steht.
- [pca-demo](https://github.com/sebastian-hanisch/pca-demo): kürzt Daten mit derselben Zerlegung; hier stabilisiert die Abschneidung eine Lösung.

## Tests

Mini-Instanz von Hand (3 × 3), abgeschnittene Lösung gegen die Pseudoinverse der Rang-k-Matrix, Eckart-Young gegen die Spektralnorm, bester Rang gegen eine unabhängige Aufzählung, Bias-Varianz gegen Monte Carlo, Freiheitsgrade gegen die Spur der Projektion, bester Filter und bester Parameter gegen scipy, Generator (SplitMix64 gegen eine unabhängige Fassung), Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Slider-Grenzen, jede Rauschstufe und Datenart, Orakel, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `tsv_model.py` | Filterlösungen, Bias-Varianz, bester Rang, bester Parameter, bester Filter, Zusatzmatrizen |
| `tsv_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `tsv_visualization.py`, `tsv_pdf_export.py` | Figuren, PDF |
| `tsv_constants.py`, `tsv_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Parameterwahl ohne Wahrheit** (L-Kurve, GCV, Diskrepanzprinzip): das Folgestück; hier sind k\* und λ\* Orakel.
- **Iterative Verfahren mit frühem Abbruch** (Landweber, CGLS): das Folgestück.
- **Weitere Filterfamilien** (exponentielle, Wiener-artige): ändern die Aussage nicht.
- **Sparsity, Total Variation, Bayes-Sicht:** spätere Stücke.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).
