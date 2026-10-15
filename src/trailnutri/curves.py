"""Courbes continues entre 0 et 1, pour remplacer les seuils durs.

Toutes acceptent un scalaire ou un tableau numpy et renvoient un tableau numpy.
"""

from __future__ import annotations

import numpy as np

Nombres = float | np.ndarray


def sigmoid(x: Nombres, centre: float, raideur: float) -> np.ndarray:
    """Sigmoïde logistique : 0.5 au centre, tend vers 1 au-dessus, vers 0 en dessous.

    `raideur` > 0 : plus elle est grande, plus la transition est brusque
    (avec raideur = 50 sur une pente, on passe de ~0.1 à ~0.9 sur ±4.4 points de pente).
    """
    if raideur <= 0:
        raise ValueError("La raideur d'une sigmoïde doit être > 0.")
    z = np.clip(raideur * (np.asarray(x, dtype=float) - centre), -500, 500)  # évite l'overflow
    return 1.0 / (1.0 + np.exp(-z))


def ramp(x: Nombres, bas: float, haut: float) -> np.ndarray:
    """Rampe linéaire bornée : 0 sous `bas`, 1 au-dessus de `haut`, linéaire entre les deux."""
    if haut <= bas:
        raise ValueError("Une rampe exige bas < haut.")
    return np.clip((np.asarray(x, dtype=float) - bas) / (haut - bas), 0.0, 1.0)


def gaussian_window(x: Nombres, centre: float, largeur: float) -> np.ndarray:
    """Fenêtre en cloche : 1 au centre, ~0.61 à ±largeur, ~0.14 à ±2 largeurs."""
    if largeur <= 0:
        raise ValueError("La largeur d'une fenêtre gaussienne doit être > 0.")
    return np.exp(-0.5 * ((np.asarray(x, dtype=float) - centre) / largeur) ** 2)
