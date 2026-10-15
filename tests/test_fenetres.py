"""Fenêtres de prises : ordre des bornes, zones interdites, resserrement, ravitos, CSV."""

from dataclasses import replace
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trailnutri.fenetres import ajouter_fenetres, texte_fenetre
from trailnutri.gpx_loader import construire_trace, lire_gpx
from trailnutri.nutrition import exporter_csv, plan_glucides
from trailnutri.pipeline import planifier, rassembler_points
from trailnutri.waypoints import Ravito, extraire_pois

EXEMPLE = Path(__file__).resolve().parents[1] / "data" / "exemple.gpx"


def segments_terrain(descente=(), montee=(), score=0.1, nb_min: int = 120) -> pd.DataFrame:
    """Un segment de 200 m par minute, difficulté constante ; portions interdites en minutes."""
    m = np.arange(nb_min)

    def entre(bornes):
        return (m >= bornes[0]) & (m < bornes[1]) if bornes else np.zeros(nb_min, dtype=bool)

    return pd.DataFrame(
        {
            "debut_m": m * 200.0,
            "fin_m": (m + 1) * 200.0,
            "temps_debut_s": m * 60.0,
            "temps_fin_s": (m + 1) * 60.0,
            "difficulte": score,
            "niveau": "facile",
            "descente_technique": entre(descente),
            "montee_dure": entre(montee),
        }
    )


def plan_aux(minutes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"temps_s": np.array(minutes) * 60.0, "glucides_g": 25,
                         "raison": "intervalle fixe"})


def fenetre(cfg, minutes, segments, **kwargs) -> pd.DataFrame:
    return ajouter_fenetres(plan_aux(minutes), segments, cfg.nutrition, cfg.decalage, **kwargs)


# --- Construction de la fenêtre ------------------------------------------------------------

def test_largeur_et_conversion_en_km(cfg):
    # Score 0.1, resserrement 1 : ± 6 x 0.9 = ± 5.4 min autour de 25 min -> 19.6 à 30.4 min
    f = fenetre(cfg, [25], segments_terrain(score=0.1)).iloc[0]
    assert f["t_min_s"] == pytest.approx(19.6 * 60)
    assert f["t_max_s"] == pytest.approx(30.4 * 60)
    assert f["km_min"] == pytest.approx(3.92)  # 200 m / min
    assert f["km_max"] == pytest.approx(6.08)
    assert f["fenetre"] == "km 4 – 6"


def test_resserrement_quand_c_est_dur(cfg):
    facile = fenetre(cfg, [25], segments_terrain(score=0.05)).iloc[0]
    dur = fenetre(cfg, [25], segments_terrain(score=0.6)).iloc[0]
    assert dur["t_max_s"] - dur["t_min_s"] < facile["t_max_s"] - facile["t_min_s"]
    sans = replace(cfg.decalage, resserrement_difficulte=0)
    dur_sans = ajouter_fenetres(plan_aux([25]), segments_terrain(score=0.6), cfg.nutrition,
                                sans).iloc[0]
    assert dur_sans["t_max_s"] - dur_sans["t_min_s"] == pytest.approx(12 * 60)


def test_s_arrete_avant_une_descente_technique(cfg):
    # Descente technique de 28 à 40 min : la fenêtre s'arrête à 28 min (km 5.6)
    f = fenetre(cfg, [25], segments_terrain(descente=(28, 40))).iloc[0]
    assert f["t_max_s"] == pytest.approx(28 * 60)
    assert f["km_max"] == pytest.approx(5.6)
    assert f["t_min_s"] < 25 * 60


def test_s_arrete_avant_une_montee_dure(cfg):
    f = fenetre(cfg, [25], segments_terrain(montee=(22, 35))).iloc[0]
    assert f["t_min_s"] == pytest.approx(25 * 60) and f["t_max_s"] == pytest.approx(25 * 60)
    f = fenetre(cfg, [21], segments_terrain(montee=(22, 35))).iloc[0]
    assert f["t_max_s"] == pytest.approx(22 * 60)


def test_ideal_interdit_fenetre_reduite_au_point(cfg):
    f = fenetre(cfg, [25], segments_terrain(descente=(20, 40))).iloc[0]
    assert f["t_min_s"] == f["t_max_s"] == 25 * 60
    assert f["fenetre"] == "vers le km 5"


def test_delai_de_depart_et_arrivee(cfg):
    segments = segments_terrain(nb_min=60)
    f = fenetre(cfg, [16, 59.5], segments)
    assert f["t_min_s"].iloc[0] >= cfg.nutrition.delai_premiere_prise_min * 60
    assert f["t_max_s"].iloc[1] == pytest.approx(59.5 * 60)  # arrivée à 60 min : pas au-delà
    assert f["t_min_s"].iloc[1] <= 59.5 * 60


def test_fenetres_voisines_sans_chevauchement(cfg):
    f = fenetre(cfg, [20, 28, 36], segments_terrain())
    assert (f["t_max_s"].iloc[:-1].to_numpy() <= f["t_min_s"].iloc[1:].to_numpy() + 1e-6).all()


def test_ravito_dans_la_fenetre(cfg):
    # Ravito au km 5.4 (= 27 min) : signalé et proposé comme point préféré
    f = fenetre(cfg, [25], segments_terrain(), ravitos=[Ravito(5.4, "Col")]).iloc[0]
    assert f["ravito_fenetre"] == "Col"
    assert "ravito « Col » (km 5.4) dans la fenêtre : point préféré" in f["raison"]
    # Ravito hors fenêtre : rien
    f = fenetre(cfg, [25], segments_terrain(), ravitos=[Ravito(9.0, "Loin")]).iloc[0]
    assert f["ravito_fenetre"] == "" and "Loin" not in f["raison"]


def test_ravito_deja_retenu_pas_de_doublon(cfg):
    plan = plan_aux([27])
    plan["raison"] = "au ravito « Col » (km 5.4)"
    f = ajouter_fenetres(plan, segments_terrain(), cfg.nutrition, cfg.decalage,
                         ravitos=[Ravito(5.4, "Col")]).iloc[0]
    assert f["raison"] == "au ravito « Col » (km 5.4)"
    assert f["ravito_fenetre"] == "Col"


def test_plan_vide(cfg):
    vide = ajouter_fenetres(plan_aux([]), segments_terrain(), cfg.nutrition, cfg.decalage)
    assert vide.empty and "km_min" in vide


@pytest.mark.parametrize(
    "km_min, km_max, arrondi, attendu",
    [
        (11.8, 13.2, 0.5, "km 12 – 13"),
        (12.1, 12.6, 0.5, "km 12 – 12.5"),
        (12.1, 12.2, 0.5, "vers le km 12"),
        (11.6, 13.4, 1.0, "km 12 – 13"),
    ],
)
def test_texte_fenetre(km_min, km_max, arrondi, attendu):
    assert texte_fenetre(km_min, km_max, arrondi) == attendu


# --- Sur la trace d'exemple, via le pipeline ------------------------------------------------

@pytest.fixture
def exemple(cfg):
    gpx = lire_gpx(EXEMPLE)
    trace = construire_trace(gpx, cfg.gpx)
    return trace, extraire_pois(gpx, trace, cfg.waypoints)


@pytest.mark.parametrize("heures, decaler", [(2, True), (2, False), (4.5, True)])
def test_exemple_bornes_et_zones_interdites(cfg, exemple, heures, decaler):
    trace, pois = exemple
    ravitos, _ = rassembler_points(cfg, pois, time(8, 0))
    res = planifier(trace, cfg, heures * 3600, ravitos, decaler=decaler)
    plan, seg = res.plan, res.segments
    assert (plan["km_min"] <= plan["km"] + 1e-9).all()
    assert (plan["km"] <= plan["km_max"] + 1e-9).all()

    interdit = (seg["descente_technique"] | seg["montee_dure"]).to_numpy()
    for _, p in plan.iterrows():
        if p["t_min_s"] == p["t_max_s"]:
            continue  # fenêtre réduite au point idéal (pas de meilleur créneau)
        chevauche = (seg["temps_debut_s"] < p["t_max_s"]) & (seg["temps_fin_s"] > p["t_min_s"])
        assert not (chevauche.to_numpy() & interdit).any()


def test_glucides_inchanges(cfg, exemple):
    trace, _ = exemple
    res = planifier(trace, cfg, 3 * 3600)
    sans_fenetres = plan_glucides(res.segments, cfg.nutrition, decalage=cfg.decalage)
    assert res.plan["glucides_g"].sum() == sans_fenetres["glucides_g"].sum()
    assert list(res.plan["glucides_g"]) == list(sans_fenetres["glucides_g"])


def test_export_csv_fenetres(cfg, exemple):
    trace, _ = exemple
    csv = exporter_csv(planifier(trace, cfg, 2 * 3600).plan)
    entete = csv.splitlines()[0].split(";")
    assert entete == ["km_min", "km_ideal", "km_max", "heure_min", "heure_max", "g_glucides",
                      "niveau_difficulte", "eau_ml", "sodium_mg", "raison"]
