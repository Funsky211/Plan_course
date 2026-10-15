"""Fixtures communes : configuration et GPX synthétiques à réponse connue."""

from __future__ import annotations

import io

import numpy as np
import pytest

from trailnutri.config import Config, charger_config
from trailnutri.synthetique import gpx_depuis_profil


@pytest.fixture
def cfg() -> Config:
    return charger_config()


def gpx_io(distances, altitudes) -> io.StringIO:
    """GPX synthétique sous forme de fichier en mémoire."""
    return io.StringIO(gpx_depuis_profil(distances, altitudes))


@pytest.fixture
def gpx_plat() -> io.StringIO:
    """10 km parfaitement plats à 500 m."""
    d = np.arange(0, 10_001, 25.0)
    return gpx_io(d, np.full(len(d), 500.0))


@pytest.fixture
def gpx_montee_descente() -> io.StringIO:
    """3 km à +10 % (300 m D+) puis 3 km à -10 %."""
    d = np.arange(0, 6_001, 25.0)
    alt = 500 + np.where(d <= 3000, 0.10 * d, 300 - 0.10 * (d - 3000))
    return gpx_io(d, alt)
