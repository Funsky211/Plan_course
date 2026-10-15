"""Génération de GPX synthétiques (tests et trace d'exemple)."""

from __future__ import annotations

from collections.abc import Sequence

import gpxpy.gpx
import numpy as np

METRES_PAR_DEGRE_LAT = 6_371_000.0 * np.pi / 180  # ~111 195 m


def gpx_depuis_profil(
    distances_m: Sequence[float],
    altitudes_m: Sequence[float | None],
    lat0: float = 45.0,
    lon0: float = 6.0,
) -> str:
    """Construit un GPX dont la trace part plein nord, aux distances et altitudes données.

    Une altitude à None produit un point sans <ele>.
    """
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name="synthetique")
    segment = gpxpy.gpx.GPXTrackSegment()
    for d, alt in zip(distances_m, altitudes_m):
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=lat0 + d / METRES_PAR_DEGRE_LAT, longitude=lon0, elevation=alt
            )
        )
    track.segments.append(segment)
    gpx.tracks.append(track)
    return gpx.to_xml()
