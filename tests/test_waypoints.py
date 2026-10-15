import io
from datetime import time

import gpxpy.gpx
import numpy as np
import pytest

from trailnutri.gpx_loader import construire_trace, lire_gpx
from trailnutri.synthetique import METRES_PAR_DEGRE_LAT, gpx_depuis_profil
from trailnutri.waypoints import (
    Ravito,
    barrieres_depuis_config,
    barrieres_depuis_pois,
    classer,
    extraire_pois,
    fusionner_ravitos,
    lire_heure_limite,
    parser_barrieres,
    parser_ravitos,
    ravitos_depuis_pois,
)


def charger(cfg, waypoints, distances=None, altitudes=None):
    """GPX synthétique (6 km plats par défaut) -> (gpx, trace)."""
    d = np.arange(0, 6_001, 25.0) if distances is None else distances
    alt = np.full(len(d), 500.0) if altitudes is None else altitudes
    gpx = lire_gpx(io.StringIO(gpx_depuis_profil(d, alt, waypoints=waypoints)))
    return gpx, construire_trace(gpx, cfg.gpx)


@pytest.mark.parametrize(
    "texte, attendu",
    [
        ("Ravito du Col", ("ravito", "col")),
        ("RAVITAILLEMENT complet", ("ravito",)),
        ("Point d’eau", ("eau",)),  # apostrophe typographique
        ("Barrière horaire 14h30 + ravito", ("ravito", "barriere")),
        ("Aid Station 3 / Water", ("ravito", "eau")),
        ("Colline du loup", ()),  # 'col' doit être un mot entier
        ("Sommet", ("sommet",)),
    ],
)
def test_classer(cfg, texte, attendu):
    assert set(classer(texte, cfg.waypoints)) == set(attendu)


@pytest.mark.parametrize(
    "texte, attendu",
    [
        ("Barrière 14h30", "14:30"),
        ("cutoff 9:05", "09:05"),
        ("Barrière 15h", "15:00"),
        ("km 12", ""),
    ],
)
def test_lire_heure_limite(texte, attendu):
    assert lire_heure_limite(texte) == attendu


def test_projection_km(cfg):
    gpx, trace = charger(cfg, [(2_000, "Ravito 1", ""), (4_500, "Barrière", "limite 10h00")])
    pois = extraire_pois(gpx, trace, cfg.waypoints)
    assert list(pois["km"]) == pytest.approx([2.0, 4.5], abs=0.02)
    assert pois.loc[1, "heure_limite"] == "10:00"


def test_poi_loin_de_la_trace(cfg):
    gpx, trace = charger(cfg, [])
    # Waypoint à ~1 km à l'est de la trace
    gpx.waypoints.append(
        gpxpy.gpx.GPXWaypoint(latitude=45.01, longitude=6.013, name="Refuge")
    )
    pois = extraire_pois(gpx, trace, cfg.waypoints)
    assert np.isnan(pois.loc[0, "km"])
    assert pois.loc[0, "ecart_trace_m"] > 900


def test_aller_retour_deux_passages(cfg):
    # Aller 3 km puis retour : le ravito du km 1 est croisé aux km 1 et 5
    aller = np.arange(0, 3_001, 25.0)
    gpx_xml = gpx_depuis_profil(
        np.concatenate([aller, 6_000 - aller[::-1][1:]]),
        np.full(2 * len(aller) - 1, 500.0),
        waypoints=[(1_000, "Ravito", "")],
    )
    # gpx_depuis_profil trace tout droit vers le nord : on replie la seconde moitié vers le sud
    gpx = lire_gpx(io.StringIO(gpx_xml))
    for p in gpx.tracks[0].segments[0].points:
        d = (p.latitude - 45.0) * METRES_PAR_DEGRE_LAT
        if d > 3_000:
            p.latitude = 45.0 + (6_000 - d) / METRES_PAR_DEGRE_LAT
    trace = construire_trace(gpx, cfg.gpx)
    pois = extraire_pois(gpx, trace, cfg.waypoints)
    assert list(pois["km"]) == pytest.approx([1.0, 5.0], abs=0.02)


def test_ravitos_et_barrieres_depuis_pois(cfg):
    gpx, trace = charger(
        cfg,
        [
            (1_500, "Fontaine", "point d'eau"),
            (3_000, "Ravito", "Barrière 10h30"),
            (5_000, "Sommet", ""),
        ],
    )
    pois = extraire_pois(gpx, trace, cfg.waypoints)
    ravitos = ravitos_depuis_pois(pois)
    assert [(r.type, round(r.km, 1)) for r in ravitos] == [("eau", 1.5), ("ravito", 3.0)]

    assert barrieres_depuis_pois(pois, heure_depart=None) == []
    (b,) = barrieres_depuis_pois(pois, heure_depart=time(8, 0))
    assert b.limite_s == 2.5 * 3600


def test_parser_saisies():
    assert [r.km for r in parser_ravitos("5.2, 10 ; 14,5")] == [5.2, 10.0, 14.5]
    assert parser_ravitos("  ") == []
    (b,) = parser_barrieres("12,5 ; 2:30\n\n")
    assert (b.km, b.limite_s) == (12.5, 9_000)
    with pytest.raises(ValueError):
        parser_barrieres("12.5 2:30")
    with pytest.raises(ValueError):
        parser_ravitos("km cinq")


def test_barrieres_depuis_config(cfg):
    from dataclasses import replace

    course = replace(cfg.course, barrieres=[{"km": 12.5, "limite": "2:30"}])
    (b,) = barrieres_depuis_config(course)
    assert (b.km, b.limite_s) == (12.5, 9_000)


def test_fusion_ravitos():
    ravitos = [Ravito(5.0, "GPX eau", "eau"), Ravito(5.1, "Saisi"), Ravito(10.0, "B")]
    fusion = fusionner_ravitos(ravitos, ecart_min_km=0.3)
    assert [(r.km, r.type) for r in fusion] == [(5.1, "ravito"), (10.0, "ravito")]
