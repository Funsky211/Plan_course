"""Chargement de config.yaml en dataclasses typées."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# config.yaml à la racine du repo (src/trailnutri/config.py -> ../../config.yaml)
CHEMIN_CONFIG_DEFAUT = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class GpxConfig:
    pas_reechantillonnage_m: float
    lissage_fenetre_m: float
    lissage_ordre: int
    distance_min_km: float


@dataclass(frozen=True)
class CourseConfig:
    ravitos_km: list[float]
    barrieres: list[dict]


@dataclass(frozen=True)
class WaypointsConfig:
    distance_max_trace_m: float
    ecart_min_passages_m: float
    fusion_ravitos_m: float
    mots_cles: dict[str, list[str]]


@dataclass(frozen=True)
class SegmentsConfig:
    longueur_m: float


@dataclass(frozen=True)
class PacingConfig:
    pente_min: float
    pente_max: float
    facteur_descente_min: float
    derive_fatigue: float
    allure_plate_min_s_km: float
    allure_plate_max_s_km: float
    pente_marche: float
    arret_ravito_s: float
    arret_eau_s: float


@dataclass(frozen=True)
class DifficulteConfig:
    poids: dict[str, float]
    montee_pente_min: float
    montee_longueur_min_m: float
    montee_tolerance_m: float
    montee_pente_max: float
    descente_pente_seuil: float
    descente_pente_max: float
    position_debut_fraction: float
    enchainement_recup_m: float
    creux_debut_h: float
    creux_fin_h: float
    seuil_zone_dure: float


@dataclass(frozen=True)
class NutritionConfig:
    glucides_g_h: float
    glucides_g_h_max: float
    intervalle_min: float
    delai_premiere_prise_min: float
    fraction_depart: float
    montee_en_charge_min: float
    max_unites_par_prise: int
    produits: dict[str, float]


@dataclass(frozen=True)
class HydratationConfig:
    temperature_c: float
    temperature_ref_c: float
    ml_h_base: float
    ml_h_par_degre: float
    ml_h_min: float
    ml_h_max: float
    sodium_mg_par_litre: float


@dataclass(frozen=True)
class DecalageConfig:
    fenetre_min: float
    pas_min: float
    ecart_min_entre_prises_min: float
    bonus_ravito: float
    bonus_avant_montee: float
    horizon_avant_montee_min: float
    bonus_terrain_facile: float
    terrain_facile_pente_min: float
    terrain_facile_pente_max: float
    penalite_montee_dure: float
    penalite_descente_technique: float
    penalite_par_min_decalage: float


@dataclass(frozen=True)
class Config:
    gpx: GpxConfig
    course: CourseConfig
    waypoints: WaypointsConfig
    segments: SegmentsConfig
    pacing: PacingConfig
    difficulte: DifficulteConfig
    nutrition: NutritionConfig
    decalage: DecalageConfig
    hydratation: HydratationConfig


def charger_config(chemin: str | Path | None = None) -> Config:
    """Lit le YAML et construit la Config. Une clé inconnue ou manquante lève une TypeError."""
    chemin = Path(chemin) if chemin else CHEMIN_CONFIG_DEFAUT
    with open(chemin, encoding="utf-8") as f:
        brut = yaml.safe_load(f)
    return Config(
        gpx=GpxConfig(**brut["gpx"]),
        course=CourseConfig(**brut["course"]),
        waypoints=WaypointsConfig(**brut["waypoints"]),
        segments=SegmentsConfig(**brut["segments"]),
        pacing=PacingConfig(**brut["pacing"]),
        difficulte=DifficulteConfig(**brut["difficulte"]),
        nutrition=NutritionConfig(**brut["nutrition"]),
        decalage=DecalageConfig(**brut["decalage"]),
        hydratation=HydratationConfig(**brut["hydratation"]),
    )
