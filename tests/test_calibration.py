"""Calibration : on simule un coureur dont on connaît les paramètres, puis on les retrouve."""

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trailnutri.calibration import CalibrationError, calibrer, mesurer_activite
from trailnutri.gpx_loader import construire_trace, lire_gpx
from trailnutri.pacing import cout_minetti
from trailnutri.segments import decouper_segments
from trailnutri.synthetique import gpx_depuis_profil

# Coureur simulé : 5:00/km à plat, plancher de descente 0.9, 12 % de dérive
ALLURE, DESCENTE, DERIVE = 300.0, 0.9, 0.12


def allure_coureur(pente: np.ndarray, position: np.ndarray) -> np.ndarray:
    facteur = np.maximum(cout_minetti(pente) / cout_minetti(0.0), DESCENTE)
    return ALLURE * facteur * (1 + DERIVE * position)


def activite(pentes: np.ndarray, longueur: float = 200) -> pd.DataFrame:
    """Segments d'une activité simulée avec leur durée réelle."""
    n = len(pentes)
    debut = np.arange(n) * longueur
    position = (debut + longueur / 2) / (n * longueur)
    return pd.DataFrame(
        {
            "debut_m": debut,
            "fin_m": debut + longueur,
            "longueur_m": longueur,
            "pente": pentes,
            "duree_reelle_s": allure_coureur(pentes, position) * longueur / 1000,
        }
    )


def profil_varie(n: int = 120, graine: int = 0) -> np.ndarray:
    """Alternance plat / montée / descente roulante, mélangée."""
    rng = np.random.default_rng(graine)
    return rng.choice([0.0, 0.01, -0.01, 0.06, 0.10, -0.06, -0.10], size=n)


def test_retrouve_les_parametres(cfg):
    calib = calibrer([activite(profil_varie())], cfg.pacing, cfg.calibration)
    assert calib.derive_fatigue == pytest.approx(DERIVE, abs=0.01)
    assert calib.facteur_descente_min == pytest.approx(DESCENTE, abs=0.01)
    assert calib.allure_plate_s_km == pytest.approx(ALLURE, rel=0.02)
    assert calib.nb_activites == 1


def test_pauses_ignorees(cfg):
    seg = activite(profil_varie())
    # Arrêt de 10 min sur trois segments plats
    plats = seg.index[seg["pente"] == 0.0][:3]
    seg.loc[plats, "duree_reelle_s"] += 600
    calib = calibrer([seg], cfg.pacing, cfg.calibration)
    assert calib.derive_fatigue == pytest.approx(DERIVE, abs=0.02)


def test_plusieurs_activites_mediane(cfg):
    activites = [activite(profil_varie(graine=g)) for g in range(3)]
    calib = calibrer(activites, cfg.pacing, cfg.calibration)
    assert calib.nb_activites == 3
    assert calib.facteur_descente_min == pytest.approx(DESCENTE, abs=0.01)


def test_parametre_non_mesurable_garde_la_config(cfg):
    # Que du plat : pas de descente pour mesurer le plancher
    calib = calibrer([activite(np.zeros(60))], cfg.pacing, cfg.calibration)
    assert calib.facteur_descente_min == cfg.pacing.facteur_descente_min
    assert calib.derive_fatigue == pytest.approx(DERIVE, abs=0.01)


def test_bornes(cfg):
    bornes = replace(cfg.calibration, derive_bornes=[0.0, 0.05])
    calib = calibrer([activite(profil_varie())], cfg.pacing, bornes)
    assert calib.derive_fatigue == 0.05


def test_activite_inexploitable(cfg):
    with pytest.raises(CalibrationError):
        calibrer([activite(np.full(30, 0.12))], cfg.pacing, cfg.calibration)  # aucun plat


def test_activite_sans_horodatage(cfg):
    seg = activite(profil_varie()).drop(columns="duree_reelle_s")
    with pytest.raises(CalibrationError, match="horodatage"):
        mesurer_activite(seg, cfg.pacing, cfg.calibration)


def test_depuis_un_gpx_horodate(cfg):
    # Bout en bout : profil -> GPX avec temps -> trace -> segments -> calibration
    pentes = np.repeat(profil_varie(n=60, graine=3), 20)  # pente constante par tranche de 200 m
    d = np.arange(0, len(pentes) * 10 + 1, 10.0)
    alt = 500 + np.concatenate([[0.0], np.cumsum(pentes * 10)])
    position = d[1:] / d[-1]
    temps = np.concatenate([[0.0], np.cumsum(allure_coureur(pentes, position) * 10 / 1000)])

    gpx = lire_gpx(io.StringIO(gpx_depuis_profil(d, alt, temps_s=temps)))
    trace = construire_trace(gpx, cfg.gpx)
    assert "temps_reel_s" in trace
    seg = decouper_segments(trace, cfg.segments)
    calib = calibrer([seg], cfg.pacing, cfg.calibration)
    # Le lissage d'altitude brouille un peu les pentes : tolérance plus large
    assert calib.derive_fatigue == pytest.approx(DERIVE, abs=0.04)
    assert calib.facteur_descente_min == pytest.approx(DESCENTE, abs=0.05)
