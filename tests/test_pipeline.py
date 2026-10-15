"""Bout en bout sur la trace d'exemple, et profil athlète."""

from dataclasses import replace
from datetime import time
from pathlib import Path

import pytest

from trailnutri.gpx_loader import construire_trace, lire_gpx
from trailnutri.pipeline import planifier, rassembler_points
from trailnutri.profil import ProfilAthlete, ProfilError
from trailnutri.waypoints import Barriere, Ravito, extraire_pois

EXEMPLE = Path(__file__).resolve().parents[1] / "data" / "exemple.gpx"


@pytest.fixture
def exemple(cfg):
    gpx = lire_gpx(EXEMPLE)
    trace = construire_trace(gpx, cfg.gpx)
    return trace, extraire_pois(gpx, trace, cfg.waypoints)


def test_rassembler_points(cfg, exemple):
    _, pois = exemple
    course = replace(cfg.course, ravitos_km=[5.1], barrieres=[{"km": 9.5, "limite": "1:40"}])
    ravitos, barrieres = rassembler_points(
        replace(cfg, course=course), pois, time(8, 0), [Ravito(14.0, "Saisi")]
    )
    # Col (GPX, km 5) et config (km 5.1) fusionnés ; fontaine, ravito final, saisi
    assert [round(r.km) for r in ravitos] == [5, 8, 12, 14]
    assert [r.type for r in ravitos] == ["ravito", "eau", "ravito", "ravito"]
    # Barrière de config (km 9.5) + barrière du GPX (10h30 -> 2:30 de course)
    assert [(round(b.km, 1), b.limite_s) for b in barrieres] == [(9.5, 6_000), (12.0, 9_000)]


def test_planifier_exemple(cfg, exemple):
    trace, pois = exemple
    ravitos, barrieres = rassembler_points(cfg, pois, time(8, 0))
    res = planifier(trace, cfg, 2 * 3600, ravitos, barrieres, time(8, 0))

    assert res.segments["duree_s"].sum() == pytest.approx(2 * 3600)
    assert res.segments["difficulte"].between(0, 1).all()
    assert len(res.plan) >= 4
    assert res.plan["temps_s"].is_monotonic_increasing
    assert res.barrieres["ok"].all()  # 2 h de course, barrière à 2h30 : tenue
    # Le ravito du km 5 est sur le chemin : au moins une prise doit s'y caler
    assert res.plan["raison"].str.contains("au ravito").any()


def test_planifier_sans_decalage(cfg, exemple):
    trace, _ = exemple
    res = planifier(trace, cfg, 2 * 3600, decaler=False)
    assert (res.plan["raison"] == "intervalle fixe").all()


def test_barriere_ratee(cfg, exemple):
    trace, _ = exemple
    res = planifier(trace, cfg, 4 * 3600, barrieres=[Barriere(12.0, 2.5 * 3600, "B")])
    assert not res.barrieres["ok"].iloc[0]


# --- Profil athlète ------------------------------------------------------------------------

def test_profil_aller_retour(cfg):
    profil = replace(ProfilAthlete.depuis_config(cfg), glucides_g_h=80, produits=["Gel"])
    assert ProfilAthlete.depuis_json(profil.vers_json()) == profil


def test_profil_applique(cfg):
    profil = replace(ProfilAthlete.depuis_config(cfg), derive_fatigue=0.1, produits=["Barre"],
                     temperature_c=28)
    applique = profil.appliquer(cfg)
    assert applique.pacing.derive_fatigue == 0.1
    assert applique.nutrition.produits == {"Barre": 40}
    assert applique.hydratation.temperature_c == 28
    assert cfg.pacing.derive_fatigue == 0.0  # la config d'origine est intacte


@pytest.mark.parametrize(
    "texte, message",
    [
        ("pas du json", "JSON invalide"),
        ("[1, 2]", "objet JSON"),
        ('{"glucides_g_h": 60}', "manquantes"),
    ],
)
def test_profil_invalide(texte, message):
    with pytest.raises(ProfilError, match=message):
        ProfilAthlete.depuis_json(texte)


def test_profil_mauvais_types(cfg):
    profil = ProfilAthlete.depuis_config(cfg).vers_json()
    with pytest.raises(ProfilError, match="nombre"):
        ProfilAthlete.depuis_json(profil.replace('"glucides_g_h": 60', '"glucides_g_h": "60"'))
    with pytest.raises(ProfilError, match="inconnues"):
        ProfilAthlete.depuis_json(profil.replace("{", '{"poids_kg": 70,', 1))
