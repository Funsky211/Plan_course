"""Chargement de config.yaml en dataclasses typées, avec contrôle de cohérence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# config.yaml à la racine du repo (src/trailnutri/config.py -> ../../config.yaml)
CHEMIN_CONFIG_DEFAUT = Path(__file__).resolve().parents[2] / "config.yaml"

# Facteurs du score de difficulté : les poids de config.yaml doivent couvrir exactement ceux-ci
FACTEURS_DIFFICULTE = ("montee", "fatigue", "position", "descente", "enchainement", "creux")


class ConfigError(ValueError):
    """config.yaml incohérent (valeur hors bornes, clé manquante...)."""


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
class CalibrationConfig:
    pente_plate_max: float
    descente_pente_min: float
    descente_pente_max: float
    facteur_pause: float
    min_segments: int
    facteur_descente_bornes: list[float]
    derive_bornes: list[float]


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
    glucides_max_par_produit_g: float
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
    calibration: CalibrationConfig
    difficulte: DifficulteConfig
    nutrition: NutritionConfig
    decalage: DecalageConfig
    hydratation: HydratationConfig


def charger_config(chemin: str | Path | None = None) -> Config:
    """Lit le YAML, construit la Config et la valide (ConfigError sinon)."""
    chemin = Path(chemin) if chemin else CHEMIN_CONFIG_DEFAUT
    with open(chemin, encoding="utf-8") as f:
        brut = yaml.safe_load(f)
    try:
        cfg = _construire(brut)
    except (KeyError, TypeError) as erreur:
        raise ConfigError(
            f"Section ou clé manquante / inconnue dans {chemin.name} : {erreur}"
        ) from erreur
    valider(cfg)
    return cfg


def valider(cfg: Config) -> None:
    """Vérifie la cohérence des paramètres ; liste toutes les erreurs d'un coup."""
    regles = [
        (cfg.gpx.pas_reechantillonnage_m > 0, "gpx.pas_reechantillonnage_m doit être > 0"),
        (cfg.gpx.lissage_fenetre_m > 0, "gpx.lissage_fenetre_m doit être > 0"),
        (cfg.gpx.lissage_ordre >= 1, "gpx.lissage_ordre doit être >= 1"),
        (cfg.segments.longueur_m > 0, "segments.longueur_m doit être > 0"),
        (cfg.pacing.pente_min < 0 < cfg.pacing.pente_max, "pacing : pente_min < 0 < pente_max"),
        (0 < cfg.pacing.facteur_descente_min <= 1, "pacing.facteur_descente_min dans ]0, 1]"),
        (cfg.pacing.derive_fatigue >= 0, "pacing.derive_fatigue doit être >= 0"),
        (0 < cfg.pacing.allure_plate_min_s_km < cfg.pacing.allure_plate_max_s_km,
         "pacing : 0 < allure_plate_min_s_km < allure_plate_max_s_km"),
        (min(cfg.pacing.arret_ravito_s, cfg.pacing.arret_eau_s) >= 0,
         "pacing : les temps d'arrêt doivent être >= 0"),
        (set(cfg.difficulte.poids) == set(FACTEURS_DIFFICULTE),
         f"difficulte.poids doit contenir exactement {FACTEURS_DIFFICULTE}"),
        (all(w >= 0 for w in cfg.difficulte.poids.values())
         and sum(cfg.difficulte.poids.values()) > 0,
         "difficulte.poids : poids >= 0 et de somme > 0"),
        (0 <= cfg.difficulte.seuil_zone_dure <= 1, "difficulte.seuil_zone_dure dans [0, 1]"),
        (cfg.difficulte.montee_pente_max > cfg.difficulte.montee_pente_min > 0,
         "difficulte : 0 < montee_pente_min < montee_pente_max"),
        (cfg.difficulte.descente_pente_max < cfg.difficulte.descente_pente_seuil < 0,
         "difficulte : descente_pente_max < descente_pente_seuil < 0"),
        (cfg.difficulte.creux_debut_h < cfg.difficulte.creux_fin_h,
         "difficulte : creux_debut_h < creux_fin_h"),
        (0 < cfg.difficulte.position_debut_fraction < 1,
         "difficulte.position_debut_fraction dans ]0, 1["),
        (cfg.difficulte.enchainement_recup_m > 0, "difficulte.enchainement_recup_m doit être > 0"),
        (0 < cfg.nutrition.glucides_g_h <= cfg.nutrition.glucides_g_h_max,
         "nutrition : 0 < glucides_g_h <= glucides_g_h_max"),
        (cfg.nutrition.intervalle_min > 0, "nutrition.intervalle_min doit être > 0"),
        (0 < cfg.nutrition.fraction_depart <= 1, "nutrition.fraction_depart dans ]0, 1]"),
        (cfg.nutrition.montee_en_charge_min >= 0, "nutrition.montee_en_charge_min doit être >= 0"),
        (cfg.nutrition.max_unites_par_prise >= 1, "nutrition.max_unites_par_prise doit être >= 1"),
        (all(0 < g <= cfg.nutrition.glucides_max_par_produit_g
             for g in cfg.nutrition.produits.values()),
         "nutrition.produits : glucides par unité dans ]0, glucides_max_par_produit_g]"),
        (cfg.decalage.fenetre_min >= 0 and cfg.decalage.pas_min > 0,
         "decalage : fenetre_min >= 0 et pas_min > 0"),
        (0 < cfg.hydratation.ml_h_min <= cfg.hydratation.ml_h_max,
         "hydratation : 0 < ml_h_min <= ml_h_max"),
        (cfg.waypoints.distance_max_trace_m > 0, "waypoints.distance_max_trace_m doit être > 0"),
        (all(cfg.waypoints.mots_cles.values()), "waypoints.mots_cles : aucune liste vide"),
        (cfg.calibration.facteur_descente_bornes[0] <= cfg.calibration.facteur_descente_bornes[1]
         and cfg.calibration.derive_bornes[0] <= cfg.calibration.derive_bornes[1],
         "calibration : bornes [min, max] avec min <= max"),
    ]
    erreurs = [message for ok, message in regles if not ok]
    if erreurs:
        raise ConfigError("Configuration invalide :\n- " + "\n- ".join(erreurs))


def _construire(brut: dict) -> Config:
    return Config(
        gpx=GpxConfig(**brut["gpx"]),
        course=CourseConfig(**brut["course"]),
        waypoints=WaypointsConfig(**brut["waypoints"]),
        segments=SegmentsConfig(**brut["segments"]),
        pacing=PacingConfig(**brut["pacing"]),
        calibration=CalibrationConfig(**brut["calibration"]),
        difficulte=DifficulteConfig(**brut["difficulte"]),
        nutrition=NutritionConfig(**brut["nutrition"]),
        decalage=DecalageConfig(**brut["decalage"]),
        hydratation=HydratationConfig(**brut["hydratation"]),
    )
