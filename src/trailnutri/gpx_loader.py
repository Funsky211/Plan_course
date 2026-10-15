"""Lecture d'un GPX : distance cumulée, rééchantillonnage régulier et lissage de l'altitude."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import gpxpy
import gpxpy.gpx
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from trailnutri.config import GpxConfig

RAYON_TERRE_M = 6_371_000.0


class GpxError(ValueError):
    """GPX inexploitable (illisible, vide, sans altitude, trop court...)."""


def lire_gpx(source: str | Path | IO) -> gpxpy.gpx.GPX:
    """Parse le fichier GPX (chemin ou fichier ouvert)."""
    try:
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as f:
                return gpxpy.parse(f)
        return gpxpy.parse(source)
    except gpxpy.gpx.GPXException as erreur:
        raise GpxError(f"Fichier GPX illisible : {erreur}") from erreur


def charger_gpx(source: str | Path | IO, cfg: GpxConfig) -> pd.DataFrame:
    """Raccourci : lire_gpx puis construire_trace."""
    return construire_trace(lire_gpx(source), cfg)


def construire_trace(gpx: gpxpy.gpx.GPX, cfg: GpxConfig) -> pd.DataFrame:
    """Transforme le GPX en trace régulière.

    Colonnes : distance_m, lat, lon, altitude_brute_m, altitude_m (lissée),
    et temps_reel_s si tous les points sont horodatés (activité enregistrée).
    Pas uniforme, au plus `cfg.pas_reechantillonnage_m` mètres.
    """
    lat, lon, ele, temps = _extraire_points(gpx)
    distance = _distance_cumulee(lat, lon)

    # On retire les points immobiles (pause montre) : np.interp exige des x strictement croissants
    garde = np.concatenate([[True], np.diff(distance) > 0])

    total = distance[garde][-1]
    if total < cfg.distance_min_km * 1000:
        raise GpxError(
            f"Trace trop courte : {total / 1000:.2f} km (minimum {cfg.distance_min_km} km)."
        )

    # Grille strictement uniforme de 0 à total (pas <= pas de config) : le lissage l'exige
    nb_points = int(np.ceil(total / cfg.pas_reechantillonnage_m)) + 1
    grille = np.linspace(0.0, total, nb_points)

    def sur_grille(valeurs: np.ndarray) -> np.ndarray:
        return np.interp(grille, distance[garde], valeurs[garde])

    altitude_brute = sur_grille(ele)
    trace = pd.DataFrame(
        {
            "distance_m": grille,
            "lat": sur_grille(lat),
            "lon": sur_grille(lon),
            "altitude_brute_m": altitude_brute,
            "altitude_m": _lisser(altitude_brute, cfg),
        }
    )
    if temps is not None:
        trace["temps_reel_s"] = sur_grille(temps)
    return trace


def denivele(altitudes: np.ndarray | pd.Series) -> tuple[float, float]:
    """Renvoie (D+, D-) en mètres, D- en valeur positive."""
    ecarts = np.diff(np.asarray(altitudes, dtype=float))
    return float(ecarts[ecarts > 0].sum()), float(-ecarts[ecarts < 0].sum())


def distance_haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distance à plat (m) entre points, vectorisée (accepte scalaires et tableaux)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * RAYON_TERRE_M * np.arcsin(np.sqrt(a))


def _extraire_points(
    gpx: gpxpy.gpx.GPX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """lat, lon, altitude et temps écoulé (ou None) de toutes les traces, à défaut des routes."""
    points = [p for t in gpx.tracks for s in t.segments for p in s.points]
    if not points:
        points = [p for r in gpx.routes for p in r.points]
    if len(points) < 2:
        raise GpxError("Le GPX ne contient pas assez de points (trace ou route).")

    lat = np.array([p.latitude for p in points], dtype=float)
    lon = np.array([p.longitude for p in points], dtype=float)
    # None -> NaN pour pouvoir détecter les altitudes manquantes
    ele = np.array([np.nan if p.elevation is None else p.elevation for p in points], dtype=float)

    manquantes = np.isnan(ele)
    if manquantes.all():
        raise GpxError("Le GPX ne contient aucune altitude : impossible de calculer les pentes.")
    if manquantes.any():
        # Quelques trous : on interpole linéairement sur l'indice du point
        idx = np.arange(len(ele))
        ele[manquantes] = np.interp(idx[manquantes], idx[~manquantes], ele[~manquantes])

    # Temps : seulement si tous les points sont horodatés (sinon trace "plan", pas une activité)
    temps = None
    if all(p.time is not None for p in points):
        debut = points[0].time
        temps = np.array([(p.time - debut).total_seconds() for p in points], dtype=float)

    return lat, lon, ele, temps


def _distance_cumulee(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance horizontale cumulée (haversine), en mètres, commençant à 0."""
    pas = distance_haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(pas)])


def _lisser(altitude: np.ndarray, cfg: GpxConfig) -> np.ndarray:
    """Lissage Savitzky-Golay : enlève le bruit GPS sans écraser les vraies bosses."""
    # Fenêtre en nombre de points, impaire, > ordre, et pas plus longue que la trace
    fenetre = int(round(cfg.lissage_fenetre_m / cfg.pas_reechantillonnage_m))
    fenetre = min(fenetre, len(altitude))
    if fenetre % 2 == 0:
        fenetre -= 1
    if fenetre <= cfg.lissage_ordre:
        return altitude.copy()
    return savgol_filter(altitude, fenetre, cfg.lissage_ordre, mode="interp")
