"""Score de difficulté continu par segment (0 à 1).

Six facteurs, chacun calculé par une courbe continue (curves.py) plutôt qu'un seuil dur,
combinés par une moyenne pondérée (poids dans config.yaml), puis lissés sur quelques
centaines de mètres pour éviter les sauts brusques entre segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trailnutri.config import FACTEURS_DIFFICULTE as FACTEURS
from trailnutri.config import DifficulteConfig
from trailnutri.curves import gaussian_window, ramp, sigmoid
from trailnutri.outils import plages


def calculer_difficulte(segments: pd.DataFrame, cfg: DifficulteConfig) -> pd.DataFrame:
    """Ajoute à la sortie de calculer_temps les colonnes :

    f_<facteur> (0-1) pour chacun des FACTEURS, difficulte_brute, difficulte (lissée),
    montee_dure et descente_technique (booléens utilisés pour placer les prises), zone_dure.
    """
    seg = segments.copy()
    pente = seg["pente"].to_numpy()
    longueur = seg["longueur_m"].to_numpy()
    milieu_m = ((seg["debut_m"] + seg["fin_m"]) / 2).to_numpy()
    heures = ((seg["temps_debut_s"] + seg["temps_fin_s"]) / 2).to_numpy() / 3600

    # Montée : raide (sigmoïde sur la pente) ET longue (rampe sur la distance montée d'affilée)
    seg["montee_continue_m"] = distance_montee_continue(pente, longueur, cfg.montee_plancher)
    seg["f_montee"] = sigmoid(pente, cfg.montee_centre, cfg.montee_raideur) * ramp(
        seg["montee_continue_m"], 0, cfg.montee_longueur_pleine_m
    )

    # Descente technique : sigmoïde sur la pente négative (-pente > -centre)
    seg["f_descente"] = sigmoid(-pente, -cfg.descente_centre, cfg.descente_raideur)

    # Fatigue (temps écoulé) et position (avancement), progressives
    seg["f_fatigue"] = ramp(heures, 0, cfg.fatigue_pleine_h)
    seg["f_position"] = ramp(milieu_m / seg["fin_m"].iloc[-1], cfg.position_debut,
                             cfg.position_fin)

    # Creux physiologique : cloche autour de ~3h15
    seg["f_creux"] = gaussian_window(heures, cfg.creux_centre_h, cfg.creux_largeur_h)

    # Booléens "à éviter pour manger", dérivés des facteurs continus
    seg["montee_dure"] = seg["f_montee"] >= cfg.seuil_contrainte
    seg["descente_technique"] = seg["f_descente"] >= cfg.seuil_contrainte

    # Enchaînement : montée qui arrive peu après la précédente
    seg["f_enchainement"] = _enchainement(seg, cfg)

    poids = np.array([cfg.poids[f] for f in FACTEURS])
    facteurs = seg[[f"f_{f}" for f in FACTEURS]].to_numpy()
    seg["difficulte_brute"] = facteurs @ poids / poids.sum()
    seg["difficulte"] = lisser(seg["difficulte_brute"], longueur, cfg.lissage_fenetre_m)
    seg["zone_dure"] = seg["difficulte"] >= cfg.seuil_dur
    return seg


def distance_montee_continue(
    pente: np.ndarray, longueur: np.ndarray, plancher: float
) -> np.ndarray:
    """Distance montée d'affilée à la fin de chaque segment (remise à 0 sous le plancher)."""
    resultat = np.zeros(len(pente))
    cumul = 0.0
    for i, (p, lg) in enumerate(zip(pente, longueur, strict=True)):
        cumul = cumul + lg if p > plancher else 0.0
        resultat[i] = cumul
    return resultat


def lisser(score: pd.Series, longueur: np.ndarray, fenetre_m: float) -> pd.Series:
    """Moyenne glissante centrée sur ~fenetre_m mètres (nombre impair de segments)."""
    n = int(round(fenetre_m / np.mean(longueur)))
    if n % 2 == 0:
        n += 1  # impair -> fenêtre centrée sur le segment
    if n <= 1:
        return score.copy()
    return score.rolling(n, center=True, min_periods=1).mean()


def zones_dures(segments: pd.DataFrame) -> list[tuple[float, float]]:
    """Intervalles (km_debut, km_fin) des segments consécutifs en zone dure."""
    return [
        (segments["debut_m"].iloc[d] / 1000, segments["fin_m"].iloc[f] / 1000)
        for d, f in plages(segments["zone_dure"].to_numpy())
    ]


def _enchainement(seg: pd.DataFrame, cfg: DifficulteConfig) -> np.ndarray:
    """Pour chaque montée dure après la première : (1 - rampe sur la récupération) x f_montee.

    Récupération = distance entre la fin de la montée précédente et le début de celle-ci.
    0 m de récupération -> 1 ; au-delà de enchainement_recup_m -> 0 ; linéaire entre les deux.
    """
    score = np.zeros(len(seg))
    montees = plages(seg["montee_dure"].to_numpy())
    for (_, fin_prec), (debut, fin) in zip(montees, montees[1:], strict=False):
        recup_m = seg["debut_m"].iloc[debut] - seg["fin_m"].iloc[fin_prec]
        intensite = 1 - float(ramp(recup_m, 0, cfg.enchainement_recup_m))
        score[debut : fin + 1] = intensite * seg["f_montee"].iloc[debut : fin + 1]
    return score
