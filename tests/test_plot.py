import numpy as np
import pandas as pd
import pytest

from trailnutri.plot import profil_altimetrique


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


def test_zones_dures(trace):
    fig = profil_altimetrique(trace, zones_dures=[(1.0, 2.0), (3.5, 4.0)])
    assert len(fig.layout.shapes) == 2
