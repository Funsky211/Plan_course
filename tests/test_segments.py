import numpy as np
import pandas as pd
import pytest

from trailnutri.config import SegmentsConfig
from trailnutri.gpx_loader import charger_gpx
from trailnutri.segments import decouper_segments


def trace_simple(total_m: float, pente: float, pas: float = 10.0) -> pd.DataFrame:
    """Trace déjà rééchantillonnée, pente constante, sans passer par le GPX."""
    d = np.append(np.arange(0, total_m, pas), total_m)
    alt = 500 + pente * d
    return pd.DataFrame({"distance_m": d, "altitude_brute_m": alt, "altitude_m": alt})


def test_longueurs_et_couverture():
    seg = decouper_segments(trace_simple(1_000, 0.0), SegmentsConfig(longueur_m=150))
    # 1000 = 6 x 150 + 100 ; le reste (100 m >= 75 m) forme un 7e segment
    assert len(seg) == 7
    assert seg["longueur_m"].sum() == pytest.approx(1_000)
    assert seg["debut_m"].iloc[0] == 0
    assert seg["fin_m"].iloc[-1] == pytest.approx(1_000)


def test_petit_reste_fusionne():
    # 920 = 6 x 150 + 20 : le reste de 20 m (< 75 m) est fusionné -> 6 segments, le dernier de 170 m
    seg = decouper_segments(trace_simple(920, 0.0), SegmentsConfig(longueur_m=150))
    assert len(seg) == 6
    assert seg["longueur_m"].iloc[-1] == pytest.approx(170)


def test_pente_constante():
    seg = decouper_segments(trace_simple(3_000, 0.08), SegmentsConfig(longueur_m=150))
    assert seg["pente"].to_numpy() == pytest.approx(0.08)
    assert seg["d_plus_m"].sum() == pytest.approx(240)
    assert seg["d_moins_m"].sum() == pytest.approx(0)
    assert seg["d_plus_cumul_m"].iloc[-1] == pytest.approx(240)


def test_depuis_gpx_montee_descente(cfg, gpx_montee_descente):
    trace = charger_gpx(gpx_montee_descente, cfg.gpx)
    seg = decouper_segments(trace, cfg.segments)
    # Loin du sommet (lissé), les pentes valent +10 % puis -10 %
    montee = seg[seg["fin_m"] < 2_500]
    descente = seg[seg["debut_m"] > 3_500]
    assert montee["pente"].to_numpy() == pytest.approx(0.10, abs=1e-3)
    assert descente["pente"].to_numpy() == pytest.approx(-0.10, abs=1e-3)
