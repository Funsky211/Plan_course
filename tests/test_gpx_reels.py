"""GPX au format de vrais exports (structure XML réelle, contenu synthétique) : tests/data/."""

from pathlib import Path

import pytest

from trailnutri.gpx_loader import GpxError, construire_trace, denivele, lire_gpx
from trailnutri.segments import decouper_segments
from trailnutri.waypoints import extraire_pois

DATA = Path(__file__).parent / "data"


def charger(cfg, nom):
    gpx = lire_gpx(DATA / nom)
    trace = construire_trace(gpx, cfg.gpx)
    return trace, extraire_pois(gpx, trace, cfg.waypoints)


def test_garmin_extensions_et_horodatage(cfg):
    trace, pois = charger(cfg, "garmin_activite.gpx")
    assert trace["distance_m"].iloc[-1] == pytest.approx(79 * 40, rel=0.01)
    assert "temps_reel_s" in trace  # activité horodatée -> utilisable pour la calibration
    # Symbole Garmin "Food" -> ravito
    assert pois.loc[0, "categories"] == ("ravito",)
    assert pois.loc[0, "km"] == pytest.approx(1.6, abs=0.05)
    seg = decouper_segments(trace, cfg.segments)
    assert seg["duree_reelle_s"].sum() == pytest.approx(trace["temps_reel_s"].iloc[-1])


def test_strava_descente(cfg):
    trace, pois = charger(cfg, "strava_activite.gpx")
    d_plus, d_moins = denivele(trace["altitude_m"])
    assert d_moins > 100 and d_plus < 20  # descente régulière à -4 %
    assert pois.empty


def test_route_organisateur_avec_bom(cfg):
    trace, pois = charger(cfg, "route_organisateur.gpx")
    assert "temps_reel_s" not in trace  # tracé de parcours : pas d'horodatage
    assert trace["distance_m"].iloc[-1] == pytest.approx(49 * 60, rel=0.01)
    ravito = pois.iloc[0]
    assert set(ravito["categories"]) >= {"ravito", "barriere", "col"}
    assert ravito["heure_limite"] == "11:15"
    assert "sommet" in pois.iloc[1]["categories"]


def test_plusieurs_segments_et_traces(cfg):
    trace, _ = charger(cfg, "plusieurs_segments.gpx")
    # Les 3 morceaux sont mis bout à bout : 90 points, 89 intervalles de 40 m
    assert trace["distance_m"].iloc[-1] == pytest.approx(89 * 40, rel=0.01)


def test_fichier_illisible():
    with pytest.raises(GpxError, match="illisible"):
        lire_gpx(DATA / "illisible.gpx")
