"""Calibration du pacing sur des activités passées (GPX horodatés de la montre).

Pour chaque activité :
1. allure de référence = médiane des segments plats de la 1re moitié ;
2. dérive de fatigue = pente de la droite (allure observée / allure prévue) selon la
   position dans la course, sur les segments plats ou en montée ;
3. plancher de descente = médiane (allure observée / allure plate) sur les descentes roulantes,
   corrigée de la fatigue.
Les segments anormalement lents (pauses) sont ignorés. On combine les activités par la médiane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trailnutri.config import CalibrationConfig, PacingConfig
from trailnutri.pacing import facteur_pente


class CalibrationError(ValueError):
    """Aucune activité exploitable."""


@dataclass(frozen=True)
class Mesure:
    """Ce qu'on a pu mesurer sur une activité (None si pas assez de segments)."""

    allure_plate_s_km: float
    derive_fatigue: float | None
    facteur_descente: float | None


@dataclass(frozen=True)
class Calibration:
    facteur_descente_min: float
    derive_fatigue: float
    allure_plate_s_km: float
    nb_activites: int


def mesurer_activite(
    segments: pd.DataFrame, pacing: PacingConfig, cfg: CalibrationConfig
) -> Mesure | None:
    """Mesures sur une activité (sortie de decouper_segments avec duree_reelle_s)."""
    if "duree_reelle_s" not in segments:
        raise CalibrationError("Activité sans horodatage : impossible de calibrer.")

    pente = segments["pente"].to_numpy()
    allure = (segments["duree_reelle_s"] / (segments["longueur_m"] / 1000)).to_numpy()
    milieu = ((segments["debut_m"] + segments["fin_m"]) / 2).to_numpy()
    position = milieu / segments["fin_m"].iloc[-1]  # 0 au départ, 1 à l'arrivée
    plat = np.abs(pente) <= cfg.pente_plate_max

    # 1. Allure de référence (plat, première moitié, peu de fatigue)
    reference = plat & (position < 0.5)
    if reference.sum() < cfg.min_segments:
        reference = plat
    if reference.sum() < cfg.min_segments:
        return None
    allure_ref = float(np.median(allure[reference]))

    # Pauses : segment beaucoup plus lent que ce que la pente explique
    attendu = allure_ref * facteur_pente(pente, pacing)
    actif = allure <= cfg.facteur_pause * attendu

    # 2. Dérive : allure observée / prévue ≈ a + b x position, sur plat et montée
    derive, allure_depart = None, allure_ref
    masque = actif & (pente >= 0)
    if masque.sum() >= cfg.min_segments:
        b, a = np.polyfit(position[masque], allure[masque] / attendu[masque], 1)
        derive = float(b / a)
        allure_depart = allure_ref * a  # allure plate "fraîche", au km 0

    # 3. Plancher de descente, corrigé de la fatigue mesurée
    descente = actif & (pente >= cfg.descente_pente_min) & (pente <= cfg.descente_pente_max)
    facteur_descente = None
    if descente.sum() >= cfg.min_segments:
        fatigue = 1 + (derive or 0.0) * position[descente]
        facteur_descente = float(np.median(allure[descente] / (allure_depart * fatigue)))

    return Mesure(allure_depart, derive, facteur_descente)


def calibrer(
    activites: list[pd.DataFrame], pacing: PacingConfig, cfg: CalibrationConfig
) -> Calibration:
    """Combine les mesures de plusieurs activités (médiane), bornées par la config.

    Un paramètre impossible à mesurer garde sa valeur actuelle de `pacing`.
    """
    mesures = [m for m in (mesurer_activite(a, pacing, cfg) for a in activites) if m]
    if not mesures:
        raise CalibrationError(
            "Aucune activité exploitable (pas assez de portions plates horodatées)."
        )

    def mediane(valeurs: list[float | None], defaut: float, bornes: list[float]) -> float:
        connues = [v for v in valeurs if v is not None]
        return float(np.clip(np.median(connues), *bornes)) if connues else defaut

    return Calibration(
        facteur_descente_min=mediane(
            [m.facteur_descente for m in mesures], pacing.facteur_descente_min,
            cfg.facteur_descente_bornes,
        ),
        derive_fatigue=mediane(
            [m.derive_fatigue for m in mesures], pacing.derive_fatigue, cfg.derive_bornes
        ),
        allure_plate_s_km=float(np.median([m.allure_plate_s_km for m in mesures])),
        nb_activites=len(mesures),
    )
