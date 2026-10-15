"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: SVD einer 96 x 96-Matrix (O(n^3), unter 20 ms); die Experimente nutzen feste Ströme mit 400 Wiederholungen.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
K_MIN, K_MAX, K_STEP, DEFAULT_K = 0, 96, 1, 33                    # Rang der abgeschnittenen SVD; Werte über n bedeuten: kein Schnitt (k = n)
SEED_MAX, DEFAULT_SEED = 999999, 35

SINGULAR_COND = 1e15                                              # ab hier ist die Matrix in doppelter Genauigkeit numerisch singulär
SHOW_RANGE = (-1.5, 2.0)                                          # Wertebereich der Rekonstruktion im Diagramm; größere Werte werden abgeschnitten

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 950000
EXP_REPS = 400
EXP_RANK_OFFSETS = (-10, -5, -3, -2, -1, 0, 1, 2, 3, 5)           # Experiment 1: Abstand vom besten Rang
EXP_TIK_FACTORS = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 10.0)  # Experiment 1: Vielfache des besten Parameters (Vergleich)
EXP_NOISE_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)                 # Experiment 2
EXP_W_LEVELS = (0.025, 0.03, 0.035, 0.04)                         # Experiment 2
EXP_N_SWEEP = (24, 32, 48, 64, 80, 96)                            # Experiment 3
OPERATORS = ("gauss", "integral", "integral2")                    # Experiment 2: Matrizen mit schnell und algebraisch fallenden Singulärwerten
OPERATOR_LABELS = {"gauss": "Gauß-Unschärfe (w = 0,03)", "integral": "Integration (σᵢ ~ 1/i)", "integral2": "doppelte Integration (σᵢ ~ 1/i²)"}

COLOR_TRUTH = "#2F6B65"
COLOR_NAIVE = "#c0392b"
COLOR_TIK = "#4c78a8"
COLOR_TSVD = "#D68A2E"
COLOR_BEST = "#8e44ad"
COLOR_GREY = "#8a8a8a"

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "k": DEFAULT_K, "seed": DEFAULT_SEED},
    "Zu wenig Komponenten": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "k": 15, "seed": DEFAULT_SEED},
    "Zu viele Komponenten": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "k": 40, "seed": DEFAULT_SEED},
    "Feineres Gitter": {"n": 96, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "k": DEFAULT_K, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Standardinstanz aus Stück 1 und 2 mit k = 33: nahe dem besten Rang, der Fehler fällt von 258 % auf rund 11 %.",
    "Zu wenig Komponenten": "k = 15: die Lösung verliert Struktur (Bias), der erwartete Fehler liegt bei 16 %.",
    "Zu viele Komponenten": "k = 40: das Rauschen der kleinen Singulärwerte kommt durch (Streuung), der erwartete Fehler liegt bei 19 %.",
    "Feineres Gitter": "n = 96 statt 48: derselbe Rang wie auf n = 48 genügt, der Fehler bleibt bei rund 10 %.",
}
