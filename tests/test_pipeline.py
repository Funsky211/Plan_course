"""Bout en bout sur la trace d'exemple, et profil athlète."""

import json
from dataclasses import replace
from datetime import time
from pathlib import Path

import pytest

from trailnutri.gpx_loader import construire_trace, lire_gpx
from trailnutri.pipeline import planifier, rassembler_points
from trailnutri.profil import ProfilAthlete, ProfilError, produits_depuis_saisie
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
    profil = replace(ProfilAthlete.depuis_config(cfg), glucides_g_h=80,
                     produits={"Gel": 25, "Barre maison": 35})
    assert ProfilAthlete.depuis_json(profil.vers_json()) == profil


def test_profil_applique(cfg):
    profil = replace(ProfilAthlete.depuis_config(cfg), derive_fatigue=0.1,
                     produits={"Barre": 40, "Compote": 18}, temperature_c=28)
    applique = profil.appliquer(cfg)
    assert applique.pacing.derive_fatigue == 0.1
    assert applique.nutrition.produits == {"Barre": 40, "Compote": 18}
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


def test_profil_ancien_format_liste_de_noms(cfg):
    # Profil enregistré avant les produits perso : une simple liste de noms du catalogue
    ancien = json.loads(ProfilAthlete.depuis_config(cfg).vers_json())
    ancien["produits"] = ["Gel", "Inconnu"]
    profil = ProfilAthlete.depuis_json(json.dumps(ancien), catalogue=cfg.nutrition.produits)
    assert profil.produits == {"Gel": 25}  # nom inconnu du catalogue ignoré


def test_profil_produits_invalides(cfg):
    profil = ProfilAthlete.depuis_config(cfg).vers_json()
    with pytest.raises(ProfilError, match="produits"):
        ProfilAthlete.depuis_json(profil.replace('"Gel": 25', '"Gel": -3'))


# --- Saisie des produits perso -------------------------------------------------------------

def test_saisie_produits_valide():
    lignes = [("Gel", 25), ("  Barre maison ", 35.0), (None, None), ("", float("nan"))]
    assert produits_depuis_saisie(lignes, 200) == {"Gel": 25.0, "Barre maison": 35.0}


@pytest.mark.parametrize(
    "lignes, message",
    [
        ([(None, 30)], "pas de nom"),
        ([("Compote", None)], "manquants"),
        ([("Compote", float("nan"))], "manquants"),
        ([("Compote", 0)], "entre 1 et 200"),
        ([("Compote", 250)], "entre 1 et 200"),
        ([("Gel", 25), ("gel", 30)], "deux fois"),  # doublon, même avec une casse différente
    ],
)
def test_saisie_produits_invalide(lignes, message):
    with pytest.raises(ProfilError, match=message):
        produits_depuis_saisie(lignes, 200)


def test_produits_perso_dans_le_plan(cfg, exemple):
    # Seulement un produit perso de 30 g : chaque prise est exprimée avec lui
    trace, _ = exemple
    profil = replace(ProfilAthlete.depuis_config(cfg), produits={"Gel maison": 30})
    res = planifier(trace, profil.appliquer(cfg), 3 * 3600, decaler=False)
    assert res.plan["a_prendre"].str.contains("Gel maison").all()
    assert set(res.plan["glucides_g"]) <= {30.0, 60.0}
