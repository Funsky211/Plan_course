"""Temps de passage par segment : allure ajustée à la pente (Minetti), recalée sur le temps cible."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trailnutri.config import PacingConfig


class PacingError(ValueError):
    """Temps cible invalide ou irréaliste."""


def cout_minetti(pente: np.ndarray | float) -> np.ndarray:
    """Coût énergétique de la course (J/kg/m) selon Minetti et al. 2002, pente en fraction."""
    i = np.asarray(pente, dtype=float)
    return 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6


def facteur_pente(pente: np.ndarray | float, cfg: PacingConfig) -> np.ndarray:
    """Combien de fois plus de temps qu'à plat pour un mètre à cette pente.

    1.0 à plat, > 1 en montée et en descente raide, plancher en descente douce.
    """
    i = np.clip(pente, cfg.pente_min, cfg.pente_max)
    facteur = cout_minetti(i) / cout_minetti(0.0)
    # Minetti seul rendrait les descentes douces bien trop rapides (technique, sécurité)
    return np.maximum(facteur, cfg.facteur_descente_min)


def calculer_temps(
    segments: pd.DataFrame, temps_cible_s: float, cfg: PacingConfig
) -> pd.DataFrame:
    """Ajoute à `segments` les colonnes de temps, total recalé sur `temps_cible_s`.

    Colonnes ajoutées : facteur_pente, facteur_fatigue, duree_s, temps_debut_s,
    temps_fin_s, allure_s_km.
    """
    if temps_cible_s <= 0:
        raise PacingError("Le temps cible doit être strictement positif.")

    seg = segments.copy()
    distance_totale = seg["fin_m"].iloc[-1]

    # Fatigue : facteur linéaire de 1 (départ) à 1 + dérive (arrivée), au milieu du segment
    milieu = (seg["debut_m"] + seg["fin_m"]) / 2
    seg["facteur_pente"] = facteur_pente(seg["pente"].to_numpy(), cfg)
    seg["facteur_fatigue"] = 1 + cfg.derive_fatigue * milieu / distance_totale

    # "Distance équivalente plat" de chaque segment ; le temps lui est proportionnel
    equiv_plat_m = seg["longueur_m"] * seg["facteur_pente"] * seg["facteur_fatigue"]
    allure_plate_s_km = temps_cible_s / (equiv_plat_m.sum() / 1000)
    _verifier_allure(allure_plate_s_km, cfg)

    seg["duree_s"] = equiv_plat_m / 1000 * allure_plate_s_km
    seg["temps_fin_s"] = seg["duree_s"].cumsum()
    seg["temps_debut_s"] = seg["temps_fin_s"] - seg["duree_s"]
    seg["allure_s_km"] = seg["duree_s"] / (seg["longueur_m"] / 1000)
    return seg


def _verifier_allure(allure_plate_s_km: float, cfg: PacingConfig) -> None:
    """Refuse un temps cible qui impliquerait une allure à plat hors des bornes humaines."""
    if not cfg.allure_plate_min_s_km <= allure_plate_s_km <= cfg.allure_plate_max_s_km:
        raise PacingError(
            f"Temps cible irréaliste : allure à plat équivalente de {_mmss(allure_plate_s_km)}/km "
            f"(attendu entre {_mmss(cfg.allure_plate_min_s_km)} et "
            f"{_mmss(cfg.allure_plate_max_s_km)}/km)."
        )


def _mmss(secondes: float) -> str:
    return f"{int(secondes // 60)}:{int(secondes % 60):02d}"
