"""Temps de passage par segment : allure ajustée à la pente (Minetti), recalée sur le temps cible.

Le temps cible inclut les arrêts aux ravitos : temps en mouvement = cible - arrêts.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from trailnutri.config import PacingConfig
from trailnutri.temps import formater_heure, formater_hmm, formater_mmss
from trailnutri.waypoints import Barriere, Ravito


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
    segments: pd.DataFrame,
    temps_cible_s: float,
    cfg: PacingConfig,
    ravitos: list[Ravito] | None = None,
) -> pd.DataFrame:
    """Ajoute à `segments` les colonnes de temps, total recalé sur `temps_cible_s`.

    Colonnes ajoutées : facteur_pente, facteur_fatigue, mode ('course' / 'marche'),
    mouvement_s, arret_s, duree_s, temps_debut_s, temps_fin_s, allure_s_km (hors arrêts).
    """
    if temps_cible_s <= 0:
        raise PacingError("Le temps cible doit être strictement positif.")

    seg = segments.copy()
    distance_totale = seg["fin_m"].iloc[-1]

    # Arrêts aux ravitos : rattachés au segment qui contient le km du ravito
    seg["arret_s"] = 0.0
    for ravito in ravitos or []:
        if 0 <= ravito.km * 1000 <= distance_totale:
            i = min(int(np.searchsorted(seg["fin_m"], ravito.km * 1000)), len(seg) - 1)
            duree = cfg.arret_ravito_s if ravito.type == "ravito" else cfg.arret_eau_s
            seg.loc[seg.index[i], "arret_s"] += duree
    temps_mouvement_s = temps_cible_s - seg["arret_s"].sum()
    if temps_mouvement_s <= 0:
        raise PacingError("Les arrêts aux ravitos dépassent le temps cible.")

    # Fatigue : facteur linéaire de 1 (départ) à 1 + dérive (arrivée), au milieu du segment
    milieu = (seg["debut_m"] + seg["fin_m"]) / 2
    seg["facteur_pente"] = facteur_pente(seg["pente"].to_numpy(), cfg)
    seg["facteur_fatigue"] = 1 + cfg.derive_fatigue * milieu / distance_totale
    seg["mode"] = np.where(seg["pente"] >= cfg.pente_marche, "marche", "course")

    # "Distance équivalente plat" de chaque segment ; le temps lui est proportionnel
    equiv_plat_m = seg["longueur_m"] * seg["facteur_pente"] * seg["facteur_fatigue"]
    allure_plate_s_km = temps_mouvement_s / (equiv_plat_m.sum() / 1000)
    _verifier_allure(allure_plate_s_km, cfg)

    seg["mouvement_s"] = equiv_plat_m / 1000 * allure_plate_s_km
    seg["duree_s"] = seg["mouvement_s"] + seg["arret_s"]
    seg["temps_fin_s"] = seg["duree_s"].cumsum()
    seg["temps_debut_s"] = seg["temps_fin_s"] - seg["duree_s"]
    seg["allure_s_km"] = seg["mouvement_s"] / (seg["longueur_m"] / 1000)
    return seg


def allure_plate(segments: pd.DataFrame) -> float:
    """Allure à plat équivalente (s/km) d'un plan calculé : le segment plat de référence."""
    return float(
        (segments["mouvement_s"] / (segments["longueur_m"] / 1000)
         / (segments["facteur_pente"] * segments["facteur_fatigue"])).iloc[0]
    )


def temps_au_km(segments: pd.DataFrame, km: float | np.ndarray) -> np.ndarray:
    """Temps écoulé (s) au passage du km donné, interpolé sur les bornes des segments."""
    temps = np.concatenate([[0.0], segments["temps_fin_s"].to_numpy()])
    distance = np.concatenate([[0.0], segments["fin_m"].to_numpy()])
    return np.interp(np.asarray(km) * 1000, distance, temps)


def passages_par_km(segments: pd.DataFrame, heure_depart: time | None = None) -> pd.DataFrame:
    """Temps de passage à chaque km plein (et à l'arrivée), avec le temps du km."""
    total_km = segments["fin_m"].iloc[-1] / 1000
    bornes = np.append(np.arange(1, np.floor(total_km) + 1), total_km)
    bornes = np.unique(np.round(bornes, 3))
    temps = temps_au_km(segments, bornes)
    temps_km = np.diff(np.concatenate([[0.0], temps]))
    return pd.DataFrame(
        {
            "km": bornes,
            "temps": [formater_hmm(t) for t in temps],
            "heure": [formater_heure(t, heure_depart) for t in temps],
            "temps_du_km": [formater_mmss(t) for t in temps_km],
        }
    )


def verifier_barrieres(segments: pd.DataFrame, barrieres: list[Barriere]) -> pd.DataFrame:
    """Passage prévu à chaque barrière horaire, marge et verdict."""
    lignes = []
    for b in sorted(barrieres, key=lambda b: b.km):
        passage = float(temps_au_km(segments, b.km))
        lignes.append(
            {
                "nom": b.nom,
                "km": b.km,
                "limite": formater_hmm(b.limite_s),
                "passage_prevu": formater_hmm(passage),
                "marge_min": round((b.limite_s - passage) / 60),
                "ok": passage <= b.limite_s,
            }
        )
    colonnes = ["nom", "km", "limite", "passage_prevu", "marge_min", "ok"]
    return pd.DataFrame(lignes, columns=colonnes)


def _verifier_allure(allure_plate_s_km: float, cfg: PacingConfig) -> None:
    """Refuse un temps cible qui impliquerait une allure à plat hors des bornes humaines."""
    if not cfg.allure_plate_min_s_km <= allure_plate_s_km <= cfg.allure_plate_max_s_km:
        raise PacingError(
            "Temps cible irréaliste : allure à plat équivalente de "
            f"{formater_mmss(allure_plate_s_km)}/km (attendu entre "
            f"{formater_mmss(cfg.allure_plate_min_s_km)} et "
            f"{formater_mmss(cfg.allure_plate_max_s_km)}/km)."
        )
