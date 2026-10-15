"""Génération de GPX synthétiques (tests et trace d'exemple)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import gpxpy.gpx
import numpy as np

METRES_PAR_DEGRE_LAT = 6_371_000.0 * np.pi / 180  # ~111 195 m
DEPART = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)


def gpx_depuis_profil(
    distances_m: Sequence[float],
    altitudes_m: Sequence[float | None],
    temps_s: Sequence[float] | None = None,
    waypoints: Sequence[tuple[float, str, str]] | None = None,
    lat0: float = 45.0,
    lon0: float = 6.0,
) -> str:
    """Construit un GPX dont la trace part plein nord, aux distances et altitudes données.

    Une altitude à None produit un point sans <ele>. `temps_s` (secondes depuis le départ)
    horodate les points, comme une activité enregistrée. `waypoints` : (distance_m, nom,
    description), placés exactement sur la trace.
    """
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name="synthetique")
    segment = gpxpy.gpx.GPXTrackSegment()
    for i, (d, alt) in enumerate(zip(distances_m, altitudes_m)):
        horodatage = DEPART + timedelta(seconds=float(temps_s[i])) if temps_s is not None else None
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=lat0 + d / METRES_PAR_DEGRE_LAT,
                longitude=lon0,
                elevation=alt,
                time=horodatage,
            )
        )
    track.segments.append(segment)
    gpx.tracks.append(track)

    for d, nom, description in waypoints or []:
        gpx.waypoints.append(
            gpxpy.gpx.GPXWaypoint(
                latitude=lat0 + d / METRES_PAR_DEGRE_LAT,
                longitude=lon0,
                name=nom,
                description=description or None,
            )
        )
    return gpx.to_xml()
