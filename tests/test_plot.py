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
            "glucides_g": [25.0, 25.0],
            "raison": ["intervalle fixe"] * 2,
        }
    )
    fig = profil_altimetrique(trace, plan)
    marqueurs = fig.data[1]
    assert list(marqueurs.x) == [1.0, 3.0]
    assert list(marqueurs.y) == pytest.approx([550.0, 650.0])


def test_plan_vide_ignore(trace):
    plan = pd.DataFrame(columns=["heure", "km", "d_plus_cumul_m", "glucides_g", "raison"])
    assert len(profil_altimetrique(trace, plan).data) == 1


def segments_niveaux(niveaux: list[str], longueur: float = 500) -> pd.DataFrame:
    """Segments de 500 m avec un niveau et un score, couvrant la trace de 5 km."""
    debut = np.arange(len(niveaux)) * longueur
    score = {"facile": 0.1, "modéré": 0.3, "dur": 0.5}
    return pd.DataFrame(
        {
            "debut_m": debut,
            "fin_m": debut + longueur,
            "difficulte": [score[n] for n in niveaux],
            "niveau": niveaux,
        }
    )


def test_profil_colore_par_niveau(trace):
    niveaux = ["facile"] * 4 + ["dur"] * 2 + ["modéré"] * 2 + ["facile"] * 2
    fig = profil_altimetrique(trace, segments=segments_niveaux(niveaux))
    # Une portion par zone (4 zones), une entrée de légende par niveau (3)
    assert [t.legendgroup for t in fig.data] == ["facile", "dur", "modéré", "facile"]
    assert sum(t.showlegend is not False for t in fig.data) == 3
    # Pas de trou : deux portions voisines partagent leur point de jonction
    assert fig.data[0].x[-1] == fig.data[1].x[0] == pytest.approx(2.0)
    # Survol : score numérique (interpolé, donc continu aux jonctions) et niveau
    milieu = len(fig.data[1].x) // 2  # km 2.5, au coeur de la zone dure
    assert fig.data[1].customdata[milieu][0] == pytest.approx(0.5)
    assert fig.data[1].customdata[milieu][1] == "dur"


def test_ravitos(trace):
    fig = profil_altimetrique(trace, ravitos=[Ravito(2.5, "Col")])
    assert len(fig.layout.shapes) == 1
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


def test_prises_avec_fenetre(trace):
    plan = pd.DataFrame(
        {
            "heure": ["0:25"], "km": [2.0], "km_min": [1.5], "km_max": [2.8],
            "fenetre": ["km 1.5 – 3"], "glucides_g": [25.0], "raison": ["portion roulante"],
        }
    )
    marqueur = profil_altimetrique(trace, plan).data[1]
    assert marqueur.error_x.arrayminus[0] == pytest.approx(0.5)  # fenêtre avant l'idéal
    assert marqueur.error_x.array[0] == pytest.approx(0.8)       # fenêtre après l'idéal
    assert marqueur.customdata[0][2] == "km 1.5 – 3"
