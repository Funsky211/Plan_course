"""Fonctions utilitaires partagées entre modules."""

from __future__ import annotations

import numpy as np


def plages(masque: np.ndarray) -> list[tuple[int, int]]:
    """Indices (début, fin inclus) des suites de True consécutifs.

    Ex. [F, T, T, F, T] -> [(1, 2), (4, 4)]
    """
    bords = np.diff(np.concatenate([[0], np.asarray(masque).astype(int), [0]]))
    debuts = np.flatnonzero(bords == 1)
    fins = np.flatnonzero(bords == -1) - 1
    return list(zip(debuts.tolist(), fins.tolist()))
