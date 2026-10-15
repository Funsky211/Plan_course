"""Lecture d'un GPX : distance cumulée, rééchantillonnage régulier et lissage de l'altitude."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import gpxpy
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from trailnutri.config import GpxConfig

RAYON_TERRE_M = 6_371_000.0


class GpxError(ValueError):
    """GPX inexploitable (vide, sans altitude, trop court...)."""


def charger_gpx(source: str | Path | IO, cfg: GpxConfig) -> pd.DataFrame:
    """Lit un GPX et renvoie une trace régulière.

    Colonnes : distance_m, altitude_brute_m, altitude_m (lissée).
    Pas uniforme, au plus `cfg.pas_reechantillonnage_m` mètres.
    """
    lat, lon, ele = _lire_points(source)
    distance = _distance_cumulee(lat, lon)

    # On retire les points immobiles (pause montre) : np.interp exige des x strictement croissants
    garde = np.concatenate([[True], np.diff(distance) > 0])
    distance, ele = distance[garde], ele[garde]

    total = distance[-1]
    if total < cfg.distance_min_km * 1000:
        raise GpxError(
            f"Trace trop courte : {total / 1000:.2f} km (minimum {cfg.distance_min_km} km)."
        )

    # Grille strictement uniforme de 0 à total (pas <= pas de config) : le lissage l'exige
    nb_points = int(np.ceil(total / cfg.pas_reechantillonnage_m)) + 1
    grille = np.linspace(0.0, total, nb_points)
    altitude_brute = np.interp(grille, distance, ele)

    return pd.DataFrame(
        {
            "distance_m": grille,
            "altitude_brute_m": altitude_brute,
            "altitude_m": _lisser(altitude_brute, cfg),
        }
    )


def denivele(altitudes: np.ndarray | pd.Series) -> tuple[float, float]:
    """Renvoie (D+, D-) en mètres, D- en valeur positive."""
    ecarts = np.diff(np.asarray(altitudes, dtype=float))
    return float(ecarts[ecarts > 0].sum()), float(-ecarts[ecarts < 0].sum())


def _lire_points(source: str | Path | IO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrait lat, lon, altitude de toutes les traces (ou à défaut des routes) du GPX."""
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    else:
        gpx = gpxpy.parse(source)

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

    return lat, lon, ele


def _distance_cumulee(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance horizontale cumulée (haversine), en mètres, commençant à 0."""
    phi1, phi2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dphi = phi2 - phi1
    dlambda = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    pas = 2 * RAYON_TERRE_M * np.arcsin(np.sqrt(a))
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
