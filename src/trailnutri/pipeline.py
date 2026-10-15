"""Assemblage du moteur : trace -> segments -> temps -> difficulté -> plan de prises.

L'interface (ou un script) n'appelle que `rassembler_points` et `planifier`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from trailnutri.config import Config
from trailnutri.difficulty import calculer_difficulte, zones
from trailnutri.fenetres import ajouter_fenetres
from trailnutri.nutrition import plan_glucides
from trailnutri.pacing import calculer_temps, verifier_barrieres
from trailnutri.segments import decouper_segments
from trailnutri.waypoints import (
    Barriere,
    Ravito,
    barrieres_depuis_config,
    barrieres_depuis_pois,
    fusionner_ravitos,
    ravitos_depuis_config,
    ravitos_depuis_pois,
)


@dataclass
class Resultat:
    segments: pd.DataFrame          # temps + difficulté par segment
    plan: pd.DataFrame              # prises de glucides / eau
    barrieres: pd.DataFrame         # contrôle des barrières horaires
    zones: pd.DataFrame             # zones de difficulté (facile / modéré / dur)
    ravitos: list[Ravito]


def rassembler_points(
    cfg: Config,
    pois: pd.DataFrame | None = None,
    heure_depart: time | None = None,
    ravitos_saisis: list[Ravito] | None = None,
    barrieres_saisies: list[Barriere] | None = None,
) -> tuple[list[Ravito], list[Barriere]]:
    """Ravitos et barrières venant de la config, du GPX (POI) et de la saisie, dédoublonnés."""
    ravitos = ravitos_depuis_config(cfg.course) + (ravitos_saisis or [])
    barrieres = barrieres_depuis_config(cfg.course) + (barrieres_saisies or [])
    if pois is not None:
        ravitos += ravitos_depuis_pois(pois)
        barrieres += barrieres_depuis_pois(pois, heure_depart)
    ravitos = fusionner_ravitos(ravitos, cfg.waypoints.fusion_ravitos_m / 1000)
    return ravitos, sorted(barrieres, key=lambda b: b.km)


def planifier(
    trace: pd.DataFrame,
    cfg: Config,
    temps_cible_s: float,
    ravitos: list[Ravito] | None = None,
    barrieres: list[Barriere] | None = None,
    heure_depart: time | None = None,
    decaler: bool = True,
) -> Resultat:
    """Calcule tout le plan pour un temps cible."""
    ravitos = ravitos or []
    segments = decouper_segments(trace, cfg.segments)
    segments = calculer_temps(segments, temps_cible_s, cfg.pacing, ravitos)
    segments = calculer_difficulte(segments, cfg.difficulte)
    plan = plan_glucides(
        segments,
        cfg.nutrition,
        heure_depart=heure_depart,
        decalage=cfg.decalage if decaler else None,
        ravitos=ravitos,
        hydratation=cfg.hydratation,
    )
    plan = ajouter_fenetres(plan, segments, cfg.nutrition, cfg.decalage, ravitos, heure_depart)
    return Resultat(
        segments=segments,
        plan=plan,
        barrieres=verifier_barrieres(segments, barrieres or []),
        zones=zones(segments),
        ravitos=ravitos,
    )
