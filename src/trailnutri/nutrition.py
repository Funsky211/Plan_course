"""Plan de prises de glucides : horaire de base à intervalle fixe, puis décalage optionnel.

Décalage : chaque prise peut bouger dans une fenêtre ± X min. Chaque instant candidat reçoit
un score (bonus ravito, avant montée, terrain facile ; pénalités montée dure, descente
technique, éloignement de l'horaire) et on garde le meilleur.

Quantités : glucides en grammes (g/h avec montée en charge en début de course), plus l'eau et
le sodium à boire depuis la prise précédente. Le coureur choisit lui-même ses produits.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from trailnutri.config import DecalageConfig, HydratationConfig, NutritionConfig
from trailnutri.outils import plages
from trailnutri.temps import formater_heure
from trailnutri.waypoints import Ravito

RAISON_INTERVALLE = "intervalle fixe"


class NutritionError(ValueError):
    """Paramètres de nutrition invalides."""


def plan_glucides(
    segments: pd.DataFrame,
    cfg: NutritionConfig,
    heure_depart: time | None = None,
    decalage: DecalageConfig | None = None,
    ravitos: list[Ravito] | None = None,
    hydratation: HydratationConfig | None = None,
) -> pd.DataFrame:
    """Plan de prises à partir de la sortie de calculer_temps (et calculer_difficulte si décalage).

    Sans `decalage` : une prise toutes les `cfg.intervalle_min` minutes.
    Avec `decalage` : prises déplacées selon le terrain et les ravitos.
    Colonnes : temps_s, decalage_min, heure, km, d_plus_cumul_m, glucides_g, raison ;
    plus eau_ml et sodium_mg si `hydratation` est fourni.
    """
    _verifier(cfg)
    instants_base = horaire_base(segments["temps_fin_s"].iloc[-1], cfg)

    if decalage is None:
        instants = list(instants_base)
        raisons = [RAISON_INTERVALLE] * len(instants)
        bases = list(instants_base)
    else:
        bases, instants, raisons = placer_prises(
            instants_base, segments, ravitos or [], cfg, decalage
        )

    instants_arr = np.array(instants, dtype=float)
    temps_bornes, distance_bornes, d_plus_bornes = _bornes(segments)

    # Chaque prise couvre l'intervalle de base écoulé : g/h (au milieu de l'intervalle) x durée
    intervalle_h = cfg.intervalle_min / 60
    glucides = [
        round(debit_glucides_g_h(t0 - cfg.intervalle_min * 30, cfg) * intervalle_h) for t0 in bases
    ]

    plan = pd.DataFrame(
        {
            "temps_s": instants_arr,
            "decalage_min": np.round((instants_arr - np.array(bases, dtype=float)) / 60),
            "heure": [formater_heure(t, heure_depart) for t in instants_arr],
            "km": np.interp(instants_arr, temps_bornes, distance_bornes) / 1000,
            "d_plus_cumul_m": np.interp(instants_arr, temps_bornes, d_plus_bornes),
            "glucides_g": glucides,
            "raison": raisons,
        }
    )
    if hydratation is not None:
        eau_ml = round(debit_eau_ml_h(hydratation) * intervalle_h, -1)  # arrondi à 10 ml
        plan["eau_ml"] = eau_ml
        plan["sodium_mg"] = round(eau_ml / 1000 * hydratation.sodium_mg_par_litre)
    return plan


def debit_glucides_g_h(t_s: float, cfg: NutritionConfig) -> float:
    """Cible g/h à l'instant t : de fraction_depart x cible au départ, jusqu'à la cible."""
    if cfg.montee_en_charge_min <= 0:
        return cfg.glucides_g_h
    progression = min(max(t_s, 0) / (cfg.montee_en_charge_min * 60), 1.0)
    return cfg.glucides_g_h * (cfg.fraction_depart + (1 - cfg.fraction_depart) * progression)


def debit_eau_ml_h(cfg: HydratationConfig) -> float:
    """Boisson par heure selon la température prévue, bornée."""
    ml_h = cfg.ml_h_base + cfg.ml_h_par_degre * (cfg.temperature_c - cfg.temperature_ref_c)
    return float(np.clip(ml_h, cfg.ml_h_min, cfg.ml_h_max))


def total_a_prevoir(plan: pd.DataFrame) -> pd.DataFrame:
    """Ce qu'il faut prévoir pour toute la course : glucides, eau et sodium."""
    lignes = [{"article": "Glucides", "quantite": int(plan["glucides_g"].sum()), "unite": "g"}]
    if "eau_ml" in plan:
        lignes.append(
            {"article": "Eau", "quantite": round(plan["eau_ml"].sum() / 1000, 1), "unite": "L"}
        )
        lignes.append(
            {"article": "Sodium", "quantite": int(plan["sodium_mg"].sum()), "unite": "mg"}
        )
    return pd.DataFrame(lignes, columns=["article", "quantite", "unite"], dtype=object)


def horaire_base(duree_totale_s: float, cfg: NutritionConfig) -> np.ndarray:
    """Instants des prises : intervalle, 2x intervalle, ... strictement avant l'arrivée,
    en retirant celles tombant dans le délai de départ."""
    intervalle_s = cfg.intervalle_min * 60
    instants = np.arange(intervalle_s, duree_totale_s, intervalle_s)
    return instants[instants >= cfg.delai_premiere_prise_min * 60]


def placer_prises(
    instants_base: np.ndarray,
    segments: pd.DataFrame,
    ravitos: list[Ravito],
    cfg: NutritionConfig,
    d: DecalageConfig,
) -> tuple[list[float], list[float], list[str]]:
    """Décale chaque prise vers le meilleur instant de sa fenêtre.

    Renvoie (instants de base conservés, instants choisis, raisons). Une prise peut être
    supprimée si aucune place n'existe (ex. trop près de la précédente et de l'arrivée).
    """
    for colonne in ("montee_dure", "descente_technique"):
        if colonne not in segments:
            raise ValueError("Le décalage nécessite la sortie de calculer_difficulte.")

    terrain = _Terrain(segments, ravitos, d)
    fin_course = segments["temps_fin_s"].iloc[-1]
    pas = d.pas_min * 60
    precedent = -np.inf

    bases, choisis, raisons = [], [], []
    for t0 in instants_base:
        # Bornes de la fenêtre : délai de départ, écart avec la prise précédente, arrivée
        bas = max(t0 - d.fenetre_min * 60, cfg.delai_premiere_prise_min * 60,
                  precedent + d.ecart_min_entre_prises_min * 60)
        haut = min(t0 + d.fenetre_min * 60, fin_course - 60)
        if bas > haut:
            continue

        candidats = set(np.arange(bas, haut + 1e-6, pas).tolist())
        candidats |= {t for t in terrain.temps_ravitos if bas <= t <= haut}
        if bas <= t0 <= haut:
            candidats.add(t0)
        # À score égal, on préfère le candidat le plus proche de l'horaire de base
        ordonnes = sorted(candidats, key=lambda t: abs(t - t0))
        evaluations = [terrain.evaluer(t, t0) for t in ordonnes]
        i = int(np.argmax([score for score, _ in evaluations]))

        bases.append(t0)
        choisis.append(ordonnes[i])
        raisons.append(evaluations[i][1])
        precedent = ordonnes[i]
    return bases, choisis, raisons


class _Terrain:
    """Ce que le terrain offre à un instant t : ravito, montée à venir, segment courant."""

    def __init__(self, segments: pd.DataFrame, ravitos: list[Ravito], d: DecalageConfig):
        self.d = d
        self.temps_fin = segments["temps_fin_s"].to_numpy()
        self.pente = segments["pente"].to_numpy()
        self.montee_dure = segments["montee_dure"].to_numpy()
        self.descente_technique = segments["descente_technique"].to_numpy()

        temps_bornes, distance_bornes, _ = _bornes(segments)
        self.ravitos = ravitos
        self.temps_ravitos = [float(np.interp(r.km * 1000, distance_bornes, temps_bornes))
                              for r in ravitos]
        # Début de chaque montée dure : (instant, km)
        self.montees = [
            (segments["temps_debut_s"].iloc[debut], segments["debut_m"].iloc[debut] / 1000)
            for debut, _ in plages(self.montee_dure)
        ]

    def evaluer(self, t: float, t0: float) -> tuple[float, str]:
        """Score d'une prise à l'instant t (horaire de base t0) et raison lisible."""
        d = self.d
        score = -d.penalite_par_min_decalage * abs(t - t0) / 60
        motif, alerte = "", ""

        for t_ravito, ravito in zip(self.temps_ravitos, self.ravitos, strict=True):
            if abs(t_ravito - t) < 30:
                score += d.bonus_ravito
                motif = f"au ravito « {ravito.nom} » (km {ravito.km:.1f})"
                break

        i = min(int(np.searchsorted(self.temps_fin, t, side="right")), len(self.temps_fin) - 1)
        if self.descente_technique[i]:
            score -= d.penalite_descente_technique
            alerte = "en descente technique"
        elif self.montee_dure[i]:
            score -= d.penalite_montee_dure
            alerte = "en pleine montée"
        else:
            km_montee = next(
                (km for debut, km in self.montees
                 if 0 <= debut - t <= d.horizon_avant_montee_min * 60),
                None,
            )
            if km_montee is not None:
                score += d.bonus_avant_montee
                motif = motif or f"avant la montée du km {km_montee:.1f}"
            elif d.terrain_facile_pente_min <= self.pente[i] <= d.terrain_facile_pente_max:
                score += d.bonus_terrain_facile
                motif = motif or "portion roulante"

        if motif:
            raison = motif
        elif alerte:
            raison, alerte = f"{alerte}, pas de meilleur créneau", ""
        else:
            raison = RAISON_INTERVALLE
        decalage = round((t - t0) / 60)
        if decalage:
            raison += f" (décalée de {decalage:+d} min)"
        if alerte:
            raison += f" ; {alerte}"  # ex. ravito situé en pleine montée
        return score, raison


# Export avec fenêtres : (colonne du plan, en-tête du CSV)
COLONNES_CSV_FENETRES = [
    ("km_min", "km_min"), ("km", "km_ideal"), ("km_max", "km_max"),
    ("heure_min", "heure_min"), ("heure_max", "heure_max"), ("glucides_g", "g_glucides"),
    ("niveau", "niveau_difficulte"), ("eau_ml", "eau_ml"), ("sodium_mg", "sodium_mg"),
    ("raison", "raison"),
]


def exporter_csv(plan: pd.DataFrame) -> str:
    """CSV lisible directement par un Excel français (séparateur ';', décimale ',').

    Si le plan a des fenêtres (fenetres.ajouter_fenetres) : km_min, km_ideal, km_max,
    heure_min, heure_max, g_glucides, niveau_difficulte, eau_ml, sodium_mg, raison.
    """
    if "km_min" in plan:
        paires = [(c, nom) for c, nom in COLONNES_CSV_FENETRES if c in plan]
        export = plan[[c for c, _ in paires]].rename(columns=dict(paires))
        export = export.round({"km_min": 2, "km_ideal": 2, "km_max": 2})
    else:
        colonnes = ["heure", "km", "d_plus_cumul_m", "glucides_g", "eau_ml", "sodium_mg",
                    "raison"]
        export = plan[[c for c in colonnes if c in plan]].round(
            {"km": 2, "d_plus_cumul_m": 0, "glucides_g": 1}
        )
    return export.to_csv(index=False, sep=";", decimal=",")


def _bornes(segments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temps, distance et D+ cumulé aux bornes des segments (départ inclus), pour interpoler."""
    zero = [0.0]
    return (
        np.concatenate([zero, segments["temps_fin_s"].to_numpy()]),
        np.concatenate([zero, segments["fin_m"].to_numpy()]),
        np.concatenate([zero, segments["d_plus_cumul_m"].to_numpy()]),
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
