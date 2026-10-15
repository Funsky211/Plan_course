"""Fenêtre de confiance autour de chaque prise : « prise recommandée entre le km 12 et 13 ».

Pour chaque prise (instant idéal déjà choisi par nutrition.py) :
1. demi-largeur = fenetre_min x (1 - resserrement x score de difficulté) : plus étroite si dur ;
2. bornée par le délai de départ, l'arrivée et le milieu avec les prises voisines ;
3. étendue minute par minute de chaque côté, en s'arrêtant avant toute portion interdite
   (descente technique, montée dure) ;
4. convertie en km et en heures ; un ravito dans la fenêtre est signalé comme point préféré.
Les quantités (glucides, eau, sodium) ne changent pas.
"""

from __future__ import annotations

import math
from datetime import time

import numpy as np
import pandas as pd

from trailnutri.config import DecalageConfig, NutritionConfig
from trailnutri.temps import formater_heure
from trailnutri.waypoints import Ravito

COLONNES = ["t_min_s", "t_max_s", "km_min", "km_max", "heure_min", "heure_max", "fenetre",
            "niveau", "ravito_fenetre"]


def ajouter_fenetres(
    plan: pd.DataFrame,
    segments: pd.DataFrame,
    nutrition: NutritionConfig,
    d: DecalageConfig,
    ravitos: list[Ravito] | None = None,
    heure_depart: time | None = None,
) -> pd.DataFrame:
    """Ajoute au plan les colonnes de COLONNES (sortie de calculer_difficulte requise)."""
    plan = plan.copy()
    if plan.empty:
        for colonne in COLONNES:
            plan[colonne] = pd.Series(dtype="object")
        return plan

    zones = _Interdits(segments)
    instants = plan["temps_s"].to_numpy()
    milieux = ((segments["temps_debut_s"] + segments["temps_fin_s"]) / 2).to_numpy()
    scores = np.interp(instants, milieux, segments["difficulte"].to_numpy())
    pas = d.pas_min * 60

    t_min, t_max = [], []
    for k, t in enumerate(instants):
        demi = d.fenetre_min * 60 * (1 - d.resserrement_difficulte * scores[k])
        voisin_avant = (instants[k - 1] + t) / 2 if k > 0 else -math.inf
        voisin_apres = (t + instants[k + 1]) / 2 if k + 1 < len(instants) else math.inf
        bas = max(t - demi, nutrition.delai_premiere_prise_min * 60, voisin_avant)
        haut = min(t + demi, zones.fin_course - 60, voisin_apres)
        t_min.append(zones.etendre(t, bas, -pas))
        t_max.append(zones.etendre(t, haut, pas))

    temps_bornes, distance_bornes = _bornes(segments)
    plan["t_min_s"], plan["t_max_s"] = t_min, t_max
    plan["km_min"] = np.interp(t_min, temps_bornes, distance_bornes) / 1000
    plan["km_max"] = np.interp(t_max, temps_bornes, distance_bornes) / 1000
    plan["heure_min"] = [formater_heure(t, heure_depart) for t in t_min]
    plan["heure_max"] = [formater_heure(t, heure_depart) for t in t_max]
    plan["fenetre"] = [
        texte_fenetre(a, b, d.arrondi_km) for a, b in zip(plan["km_min"], plan["km_max"],
                                                           strict=True)
    ]
    plan["niveau"] = [zones.niveau_a(t) for t in instants]
    _signaler_ravitos(plan, ravitos or [], temps_bornes, distance_bornes)
    return plan


def texte_fenetre(km_min: float, km_max: float, arrondi: float) -> str:
    """'km 12 – 13' (bornes arrondies au pas donné) ou 'vers le km 12' si elles coïncident."""
    a = round(km_min / arrondi) * arrondi
    b = round(km_max / arrondi) * arrondi
    if a == b:
        return f"vers le km {a:g}"
    return f"km {a:g} – {b:g}"


class _Interdits:
    """Portions où l'on ne place pas de prise : descente technique et montée dure."""

    def __init__(self, segments: pd.DataFrame):
        self.debut = segments["temps_debut_s"].to_numpy()
        self.fin = segments["temps_fin_s"].to_numpy()
        self.interdit = (segments["descente_technique"] | segments["montee_dure"]).to_numpy()
        self.niveaux = segments["niveau"].to_numpy()
        self.fin_course = float(self.fin[-1])

    def touche(self, a: float, b: float) -> bool:
        """Vrai si l'intervalle de temps ]a, b[ chevauche une portion interdite."""
        lo, hi = min(a, b), max(a, b)
        return bool(np.any(self.interdit & (self.debut < hi) & (self.fin > lo)))

    def etendre(self, t: float, limite: float, pas: float) -> float:
        """Étend depuis t vers la limite (pas < 0 : vers l'avant de la course), sans entrer
        dans une portion interdite. Si t est lui-même interdit, la fenêtre reste réduite à t."""
        # Limite du mauvais côté (ex. prise à moins d'une minute de l'arrivée) ou t interdit
        if (limite - t) * pas <= 0 or self.touche(t - 1e-6, t + 1e-6):
            return t
        courant = t
        while courant != limite:
            suivant = max(courant + pas, limite) if pas < 0 else min(courant + pas, limite)
            if self.touche(courant, suivant):
                break
            courant = suivant
        return courant

    def niveau_a(self, t: float) -> str:
        i = min(int(np.searchsorted(self.fin, t, side="right")), len(self.fin) - 1)
        return str(self.niveaux[i])


def _signaler_ravitos(
    plan: pd.DataFrame, ravitos: list[Ravito], temps_bornes: np.ndarray,
    distance_bornes: np.ndarray,
) -> None:
    """Ravito officiel dans la fenêtre : nommé, et proposé comme point préféré dans la raison."""
    trouves, raisons = [], []
    for t_min, t_max, raison in zip(plan["t_min_s"], plan["t_max_s"], plan["raison"],
                                    strict=True):
        ravito = next(
            (r for r in ravitos
             if t_min <= np.interp(r.km * 1000, distance_bornes, temps_bornes) <= t_max),
            None,
        )
        trouves.append(ravito.nom if ravito else "")
        if ravito and "au ravito" not in raison:
            raison += f" ; ravito « {ravito.nom} » (km {ravito.km:.1f}) dans la fenêtre : " \
                      "point préféré"
        raisons.append(raison)
    plan["ravito_fenetre"] = trouves
    plan["raison"] = raisons


def _bornes(segments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Temps et distance aux bornes des segments (départ inclus), pour interpoler."""
    return (
        np.concatenate([[0.0], segments["temps_fin_s"].to_numpy()]),
        np.concatenate([[0.0], segments["fin_m"].to_numpy()]),
    )
