"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 und 2 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1), Wahrheit x = (1, 2, 1), Daten b = A x = (1,3, 1,4, 1,3).
    Wegen x = 4/3 (1,1,1) - 1/3 (1,-2,1) (die Komponente zum Singulärwert 0,5 ist null) gilt von Hand: k = 2 liefert 4/3 (1,1,1) = (1,3333, 1,3333, 1,3333) mit Bias
    1/3 (1,-2,1), Norm 0,8165 = 33,3 % von ||x|| = 2,4495, und Varianz sigma^2 (1/1 + 1/0,25) = 5 sigma^2; k = 1 liefert dieselbe Lösung mit Varianz sigma^2;
    k = 3 ist die direkte Inversion mit Varianz 105 sigma^2 und ohne Bias."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b
