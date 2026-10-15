import numpy as np
import pytest

from conftest import gpx_io
from trailnutri.gpx_loader import GpxError, charger_gpx, denivele


def test_distance_et_pas_regulier(cfg, gpx_plat):
    trace = charger_gpx(gpx_plat, cfg.gpx)
    assert trace["distance_m"].iloc[-1] == pytest.approx(10_000, rel=1e-3)
    pas = np.diff(trace["distance_m"].to_numpy())
    assert np.allclose(pas, pas[0])  # pas uniforme
    assert pas[0] <= cfg.gpx.pas_reechantillonnage_m


def test_denivele_montee_descente(cfg, gpx_montee_descente):
    trace = charger_gpx(gpx_montee_descente, cfg.gpx)
    d_plus, d_moins = denivele(trace["altitude_m"])
    # Le lissage arrondit légèrement le sommet : tolérance de quelques mètres
    assert d_plus == pytest.approx(300, abs=5)
    assert d_moins == pytest.approx(300, abs=5)


def test_lissage_reduit_le_bruit(cfg):
    rng = np.random.default_rng(0)
    d = np.arange(0, 5_001, 10.0)
    alt = 500 + rng.normal(0, 3, len(d))  # terrain plat + bruit GPS
    trace = charger_gpx(gpx_io(d, alt), cfg.gpx)
    d_plus_brut, _ = denivele(trace["altitude_brute_m"])
    d_plus_lisse, _ = denivele(trace["altitude_m"])
    assert d_plus_lisse < d_plus_brut / 3


def test_gpx_sans_altitude(cfg):
    d = np.arange(0, 5_001, 25.0)
    with pytest.raises(GpxError, match="altitude"):
        charger_gpx(gpx_io(d, [None] * len(d)), cfg.gpx)


def test_altitudes_partielles_interpolees(cfg):
    d = np.arange(0, 5_001, 25.0)
    alt = [None if i % 5 == 0 and 0 < i < len(d) - 1 else 500.0 for i in range(len(d))]
    trace = charger_gpx(gpx_io(d, alt), cfg.gpx)
    assert trace["altitude_m"].notna().all()
    assert trace["altitude_m"].to_numpy() == pytest.approx(500.0)


def test_trace_trop_courte(cfg):
    d = np.arange(0, 501, 25.0)
    with pytest.raises(GpxError, match="trop courte"):
        charger_gpx(gpx_io(d, np.full(len(d), 500.0)), cfg.gpx)


def test_gpx_vide(cfg):
    with pytest.raises(GpxError):
        charger_gpx(gpx_io([], []), cfg.gpx)


def test_points_immobiles_ignores(cfg):
    # Une pause : le même point répété 10 fois ne doit pas casser l'interpolation
    d = np.concatenate([np.arange(0, 2_000, 25.0), [2_000.0] * 10, np.arange(2_025, 4_001, 25.0)])
    trace = charger_gpx(gpx_io(d, np.full(len(d), 500.0)), cfg.gpx)
    assert trace["distance_m"].iloc[-1] == pytest.approx(4_000, rel=1e-3)
