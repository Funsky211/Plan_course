import numpy as np
import pandas as pd
import pytest

from trailnutri.plot import graphique_allure, profil_altimetrique
from trailnutri.waypoints import Ravito


@pytest.fixture
def trace() -> pd.DataFrame:
    d = np.linspace(0, 5_000, 501)
    alt = 500 + 0.05 * d
    return pd.DataFrame({"distance_m": d, "altitude_brute_m": alt, "altitude_m": alt})


def test_profil_seul(trace):
    fig = profil_altimetrique(trace)
    assert len(fig.data) == 1
    assert fig.data[0].name == "Altitude"


def test_marqueurs_poses_sur_le_profil(trace):
    plan = pd.DataFrame(
        {
            "heure": ["0:25", "0:50"],
            "km": [1.0, 3.0],
            "d_plus_cumul_m": [50.0, 150.0],
            "a_prendre": ["1 Gel", "1 Gel"],
            "glucides_g": [25.0, 25.0],
            "raison": ["intervalle fixe"] * 2,
        }
    )
    fig = profil_altimetrique(trace, plan)
    marqueurs = fig.data[1]
    assert list(marqueurs.x) == [1.0, 3.0]
    assert list(marqueurs.y) == pytest.approx([550.0, 650.0])


def test_plan_vide_ignore(trace):
    plan = pd.DataFrame(columns=["heure", "km", "d_plus_cumul_m", "a_prendre", "raison"])
    assert len(profil_altimetrique(trace, plan).data) == 1


def test_zones_dures(trace):
    fig = profil_altimetrique(trace, zones_dures=[(1.0, 2.0), (3.5, 4.0)])
    assert len(fig.layout.shapes) == 2


def test_ravitos_et_legende_zones(trace):
    fig = profil_altimetrique(trace, zones_dures=[(1.0, 2.0)], ravitos=[Ravito(2.5, "Col")])
    assert "Zone dure" in [t.name for t in fig.data]
    assert len(fig.layout.shapes) == 2  # la zone + la ligne du ravito
    assert fig.layout.annotations[0].text == "Col"


def test_graphique_allure():
    segments = pd.DataFrame(
        {
            "debut_m": [0.0, 150.0, 300.0],
            "allure_s_km": [300.0, 900.0, 280.0],
            "pente": [0.0, 0.20, -0.05],
            "mode": ["course", "marche", "course"],
        }
    )
    fig = graphique_allure(segments)
    assert list(fig.data[0].y) == pytest.approx([5.0, 15.0, 280 / 60])
    assert list(fig.data[1].x) == [0.15]  # seul le segment en marche est marqué
    assert fig.layout.yaxis.autorange == "reversed"
