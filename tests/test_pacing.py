from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd
import pytest

from trailnutri.pacing import (
    PacingError,
    allure_plate,
    calculer_temps,
    cout_minetti,
    facteur_pente,
    passages_par_km,
    verifier_barrieres,
)
from trailnutri.waypoints import Barriere, Ravito


def segments_test(pentes: list[float], longueur: float = 1_000) -> pd.DataFrame:
    """Segments minimalistes : seules les colonnes utilisées par le pacing."""
    debut = np.arange(len(pentes)) * longueur
    return pd.DataFrame(
        {
            "debut_m": debut,
            "fin_m": debut + longueur,
            "longueur_m": longueur,
            "pente": pentes,
        }
    )


def test_minetti_plat():
    assert cout_minetti(0.0) == pytest.approx(3.6)


def test_facteur_pente(cfg):
    assert facteur_pente(0.0, cfg.pacing) == pytest.approx(1.0)
    assert facteur_pente(0.10, cfg.pacing) > 1.5          # montée nettement plus lente
    # Descente douce : plancher de config, pas plus rapide que 1/0.8 x le plat
    assert facteur_pente(-0.10, cfg.pacing) == pytest.approx(cfg.pacing.facteur_descente_min)
    # Descente très raide : de nouveau plus lente que le plat
    assert facteur_pente(-0.45, cfg.pacing) > 1.0


def test_pente_bornee(cfg):
    assert facteur_pente(0.90, cfg.pacing) == pytest.approx(facteur_pente(0.45, cfg.pacing))


def test_total_egal_temps_cible(cfg):
    seg = calculer_temps(segments_test([0.0, 0.10, -0.05, 0.02]), 1_800, cfg.pacing)
    assert seg["duree_s"].sum() == pytest.approx(1_800)
    assert seg["temps_fin_s"].iloc[-1] == pytest.approx(1_800)
    assert seg["temps_debut_s"].iloc[0] == 0


def test_plat_allure_constante(cfg):
    # 10 km plats en 50 min -> 5:00/km partout
    seg = calculer_temps(segments_test([0.0] * 10), 3_000, cfg.pacing)
    assert seg["allure_s_km"].to_numpy() == pytest.approx(300)


def test_derive_fatigue(cfg):
    pacing = replace(cfg.pacing, derive_fatigue=0.10)
    seg = calculer_temps(segments_test([0.0] * 10), 3_000, pacing)
    assert seg["duree_s"].sum() == pytest.approx(3_000)
    assert seg["allure_s_km"].is_monotonic_increasing
    assert seg["allure_s_km"].iloc[-1] > seg["allure_s_km"].iloc[0]


@pytest.mark.parametrize("temps_s", [0, -10, 600, 10 * 3_600])
def test_temps_cible_invalide(cfg, temps_s):
    # 10 km plats : 10 min (1:00/km) ou 10 h (60:00/km) sont irréalistes
    with pytest.raises(PacingError):
        calculer_temps(segments_test([0.0] * 10), temps_s, cfg.pacing)


# --- Arrêts, marche, passages, barrières ---------------------------------------------------

def test_arrets_ravito(cfg):
    ravitos = [Ravito(2.5, "R1"), Ravito(7.0, "Fontaine", "eau")]
    seg = calculer_temps(segments_test([0.0] * 10), 3_000, cfg.pacing, ravitos=ravitos)
    # Total toujours égal au temps cible, arrêts compris
    assert seg["duree_s"].sum() == pytest.approx(3_000)
    assert seg["arret_s"].sum() == cfg.pacing.arret_ravito_s + cfg.pacing.arret_eau_s
    assert seg.loc[2, "arret_s"] == cfg.pacing.arret_ravito_s  # km 2.5 -> 3e segment
    assert seg.loc[6, "arret_s"] == cfg.pacing.arret_eau_s     # km 7.0 -> fin du 7e segment
    # L'allure ne compte que le mouvement : (3000 - 150) s sur 10 km plats
    assert seg["allure_s_km"].to_numpy() == pytest.approx(285)
    assert allure_plate(seg) == pytest.approx(285)


def test_ravito_hors_trace_ignore(cfg):
    seg = calculer_temps(segments_test([0.0] * 10), 3_000, cfg.pacing, ravitos=[Ravito(42, "X")])
    assert seg["arret_s"].sum() == 0


def test_arrets_superieurs_au_temps_cible(cfg):
    ravitos = [Ravito(k, f"R{k}") for k in range(1, 10)]  # 9 x 2 min = 18 min
    with pytest.raises(PacingError, match="arrêts"):
        calculer_temps(segments_test([0.0] * 10), 15 * 60, cfg.pacing, ravitos=ravitos)


def test_mode_marche(cfg):
    seg = calculer_temps(segments_test([0.0, 0.10, 0.20]), 1_800, cfg.pacing)
    assert list(seg["mode"]) == ["course", "course", "marche"]


def test_passages_par_km(cfg):
    seg = calculer_temps(segments_test([0.0] * 3, longueur=1_500), 1_350, cfg.pacing)
    # 4.5 km à 5:00/km : passages aux km 1, 2, 3, 4 puis à l'arrivée (4.5)
    passages = passages_par_km(seg, heure_depart=time(9, 0))
    assert list(passages["km"]) == [1, 2, 3, 4, 4.5]
    assert list(passages["temps"])[:2] == ["0:05", "0:10"]
    assert passages["heure"].iloc[-1] == "09:22"  # arrivée à 9:22:30, heure tronquée
    assert passages["temps_du_km"].iloc[0] == "5:00"
    assert passages["temps_du_km"].iloc[-1] == "2:30"


def test_barrieres(cfg):
    seg = calculer_temps(segments_test([0.0] * 10), 3_000, cfg.pacing)  # 5:00/km
    controle = verifier_barrieres(
        seg, [Barriere(8.0, 35 * 60, "B2"), Barriere(4.0, 25 * 60, "B1")]
    )
    assert list(controle["nom"]) == ["B1", "B2"]  # triées par km
    # km 4 en 20 min (limite 25) : ok ; km 8 en 40 min (limite 35) : ratée de 5 min
    assert list(controle["ok"]) == [True, False]
    assert list(controle["marge_min"]) == [5, -5]
    assert list(controle["passage_prevu"]) == ["0:20", "0:40"]
