from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trailnutri.pacing import PacingError, calculer_temps, cout_minetti, facteur_pente


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
