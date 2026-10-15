"""Génère data/exemple.gpx : un trail synthétique d'environ 15 km et 800 m D+.

Usage : python scripts/generer_gpx_exemple.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trailnutri.synthetique import gpx_depuis_profil

SORTIE = Path(__file__).resolve().parents[1] / "data" / "exemple.gpx"


def profil_exemple() -> tuple[np.ndarray, np.ndarray]:
    """Profil : montée longue, crête vallonnée, descente technique, petite bosse finale."""
    rng = np.random.default_rng(42)  # graine fixe : fichier reproductible
    d = np.arange(0, 15_000, 20.0)  # un point tous les 20 m
    km = d / 1000

    alt = 500 + np.interp(
        km,
        [0, 1, 5, 8, 9.5, 12, 13.5, 15],
        [0, 0, 550, 500, 550, 100, 180, 0],
    )
    alt += 15 * np.sin(km * 3)          # petites ondulations
    alt += rng.normal(0, 1.5, len(d))   # bruit GPS
    return d, alt


if __name__ == "__main__":
    distances, altitudes = profil_exemple()
    SORTIE.parent.mkdir(exist_ok=True)
    SORTIE.write_text(gpx_depuis_profil(distances, altitudes), encoding="utf-8")
    print(f"Écrit : {SORTIE}")
