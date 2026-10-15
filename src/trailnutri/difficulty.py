"""Score de difficulté par segment (0 à 1) et zones dures.

Six facteurs normalisés entre 0 et 1, combinés par une somme pondérée (poids dans config.yaml).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trailnutri.config import FACTEURS_DIFFICULTE as FACTEURS
from trailnutri.config import DifficulteConfig
from trailnutri.outils import plages


def calculer_difficulte(segments: pd.DataFrame, cfg: DifficulteConfig) -> pd.DataFrame:
    """Ajoute à la sortie de calculer_temps les colonnes :

    f_<facteur> pour chacun des FACTEURS, difficulte (0-1), montee_dure, descente_technique
    et zone_dure (booléens).
    """
    seg = segments.copy()
    pente = seg["pente"].to_numpy()
    milieu_m = ((seg["debut_m"] + seg["fin_m"]) / 2).to_numpy()
    milieu_s = ((seg["temps_debut_s"] + seg["temps_fin_s"]) / 2).to_numpy()
    distance_totale = seg["fin_m"].iloc[-1]
    duree_totale = seg["temps_fin_s"].iloc[-1]

    # Montée soutenue : uniquement dans un bloc de montée assez long ; score selon la pente
    blocs = blocs_montee(seg, cfg)
    montee_dure = np.zeros(len(seg), dtype=bool)
    for debut, fin in blocs:
        montee_dure[debut : fin + 1] = True
    seg["montee_dure"] = montee_dure
    seg["f_montee"] = np.where(montee_dure, np.clip(pente / cfg.montee_pente_max, 0, 1), 0.0)

    # Fatigue cumulée : croît linéairement avec le temps écoulé
    seg["f_fatigue"] = milieu_s / duree_totale

    # Position : le dernier tiers pèse plus
    seg["f_position"] = (milieu_m / distance_totale >= cfg.position_debut_fraction).astype(float)

    # Descente technique : sous le seuil, score selon la pente
    seg["descente_technique"] = pente < cfg.descente_pente_seuil
    seg["f_descente"] = np.where(
        seg["descente_technique"], np.clip(pente / cfg.descente_pente_max, 0, 1), 0.0
    )

    # Enchaînement : une montée dure qui arrive peu après la précédente compte plus
    seg["f_enchainement"] = _enchainement(seg, blocs, cfg)

    # Creux physiologique : fenêtre de temps écoulé
    heures = milieu_s / 3600
    seg["f_creux"] = ((heures >= cfg.creux_debut_h) & (heures <= cfg.creux_fin_h)).astype(float)

    poids = np.array([cfg.poids[f] for f in FACTEURS])
    facteurs = seg[[f"f_{f}" for f in FACTEURS]].to_numpy()
    seg["difficulte"] = facteurs @ poids / poids.sum()
    seg["zone_dure"] = seg["difficulte"] >= cfg.seuil_zone_dure
    return seg


def blocs_montee(segments: pd.DataFrame, cfg: DifficulteConfig) -> list[tuple[int, int]]:
    """Indices (début, fin inclus) des montées soutenues.

    1. suites de segments au-dessus de montee_pente_min ;
    2. fusion de deux suites séparées par un replat <= montee_tolerance_m ;
    3. on ne garde que les blocs d'au moins montee_longueur_min_m.
    """
    debut_m = segments["debut_m"].to_numpy()
    fin_m = segments["fin_m"].to_numpy()

    fusionnes: list[tuple[int, int]] = []
    for debut, fin in plages(segments["pente"].to_numpy() > cfg.montee_pente_min):
        if fusionnes and debut_m[debut] - fin_m[fusionnes[-1][1]] <= cfg.montee_tolerance_m:
            fusionnes[-1] = (fusionnes[-1][0], fin)
        else:
            fusionnes.append((debut, fin))

    return [(d, f) for d, f in fusionnes if fin_m[f] - debut_m[d] >= cfg.montee_longueur_min_m]


def zones_dures(segments: pd.DataFrame) -> list[tuple[float, float]]:
    """Intervalles (km_debut, km_fin) des segments consécutifs en zone dure."""
    return [
        (segments["debut_m"].iloc[d] / 1000, segments["fin_m"].iloc[f] / 1000)
        for d, f in plages(segments["zone_dure"].to_numpy())
    ]


def _enchainement(
    seg: pd.DataFrame, blocs: list[tuple[int, int]], cfg: DifficulteConfig
) -> np.ndarray:
    """1 si la montée suit immédiatement la précédente, 0 si la récupération est suffisante."""
    score = np.zeros(len(seg))
    for (_, fin_prec), (debut, fin) in zip(blocs, blocs[1:], strict=False):
        recup_m = seg["debut_m"].iloc[debut] - seg["fin_m"].iloc[fin_prec]
        score[debut : fin + 1] = max(0.0, 1 - recup_m / cfg.enchainement_recup_m)
    return score
