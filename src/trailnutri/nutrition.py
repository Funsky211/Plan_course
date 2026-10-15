"""Plan de prises de glucides (MVP : intervalle fixe)."""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from trailnutri.config import NutritionConfig

RAISON_INTERVALLE = "intervalle fixe"


class NutritionError(ValueError):
    """Paramètres de nutrition invalides."""


def plan_glucides(
    segments: pd.DataFrame, cfg: NutritionConfig, heure_depart: time | None = None
) -> pd.DataFrame:
    """Une prise toutes les `cfg.intervalle_min` minutes, à partir de la sortie de calculer_temps.

    Colonnes : temps_s, heure, km, d_plus_cumul_m, glucides_g, raison.
    `heure` est le temps écoulé (h:mm), ou l'heure réelle si `heure_depart` est fourni.
    """
    _verifier(cfg)
    duree_totale_s = segments["temps_fin_s"].iloc[-1]
    intervalle_s = cfg.intervalle_min * 60

    # Instants des prises : intervalle, 2x intervalle, ... strictement avant l'arrivée,
    # en retirant celles tombant dans le délai de départ
    instants = np.arange(intervalle_s, duree_totale_s, intervalle_s)
    instants = instants[instants >= cfg.delai_premiere_prise_min * 60]

    # Position à chaque instant : interpolation sur les bornes des segments (départ inclus)
    temps_bornes = np.concatenate([[0.0], segments["temps_fin_s"].to_numpy()])
    distance_bornes = np.concatenate([[0.0], segments["fin_m"].to_numpy()])
    d_plus_bornes = np.concatenate([[0.0], segments["d_plus_cumul_m"].to_numpy()])

    # Chaque prise couvre l'intervalle écoulé : quantité = g/h x durée de l'intervalle
    glucides_par_prise = cfg.glucides_g_h * cfg.intervalle_min / 60

    return pd.DataFrame(
        {
            "temps_s": instants,
            "heure": [_formater_heure(t, heure_depart) for t in instants],
            "km": np.interp(instants, temps_bornes, distance_bornes) / 1000,
            "d_plus_cumul_m": np.interp(instants, temps_bornes, d_plus_bornes),
            "glucides_g": glucides_par_prise,
            "raison": RAISON_INTERVALLE,
        }
    )


def exporter_csv(plan: pd.DataFrame) -> str:
    """CSV lisible directement par un Excel français (séparateur ';', décimale ',')."""
    colonnes = ["heure", "km", "d_plus_cumul_m", "glucides_g", "raison"]
    return plan[colonnes].round({"km": 2, "d_plus_cumul_m": 0, "glucides_g": 1}).to_csv(
        index=False, sep=";", decimal=","
    )


def _verifier(cfg: NutritionConfig) -> None:
    if not 0 < cfg.glucides_g_h <= cfg.glucides_g_h_max:
        raise NutritionError(
            f"Cible de glucides invalide : {cfg.glucides_g_h} g/h "
            f"(attendu entre 0 et {cfg.glucides_g_h_max} g/h)."
        )
    if cfg.intervalle_min <= 0:
        raise NutritionError("L'intervalle entre prises doit être strictement positif.")
    if cfg.delai_premiere_prise_min < 0:
        raise NutritionError("Le délai avant la première prise ne peut pas être négatif.")


def _formater_heure(secondes: float, heure_depart: time | None) -> str:
    """'1:25' (temps écoulé) ou '08:25' (heure réelle si un départ est donné)."""
    if heure_depart is None:
        minutes = int(round(secondes / 60))
        return f"{minutes // 60}:{minutes % 60:02d}"
    instant = datetime.combine(datetime.today(), heure_depart) + timedelta(seconds=secondes)
    return instant.strftime("%H:%M")
