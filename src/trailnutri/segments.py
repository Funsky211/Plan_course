"""Découpage de la trace en segments de longueur fixe et calcul de la pente."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trailnutri.config import SegmentsConfig


def decouper_segments(trace: pd.DataFrame, cfg: SegmentsConfig) -> pd.DataFrame:
    """Découpe la trace (sortie de charger_gpx) en segments d'environ `cfg.longueur_m`.

    Colonnes : segment, debut_m, fin_m, longueur_m, alt_debut_m, alt_fin_m,
    d_plus_m, d_moins_m, pente (fraction, 0.10 = 10 %), d_plus_cumul_m.
    """
    distance = trace["distance_m"].to_numpy()
    altitude = trace["altitude_m"].to_numpy()
    total = distance[-1]

    # Bornes : 0, L, 2L, ..., total. Un dernier bout < L/2 est fusionné avec le précédent.
    bornes = np.arange(0.0, total, cfg.longueur_m)
    if len(bornes) > 1 and total - bornes[-1] < cfg.longueur_m / 2:
        bornes = bornes[:-1]
    bornes = np.append(bornes, total)

    # D+/D- : chaque petit pas de la trace est rattaché au segment qui contient son milieu
    ecarts = np.diff(altitude)
    milieux = (distance[:-1] + distance[1:]) / 2
    num_segment = np.searchsorted(bornes, milieux, side="right") - 1
    nb = len(bornes) - 1
    num_segment = np.clip(num_segment, 0, nb - 1)
    d_plus = np.bincount(num_segment, weights=np.clip(ecarts, 0, None), minlength=nb)
    d_moins = np.bincount(num_segment, weights=np.clip(-ecarts, 0, None), minlength=nb)

    # Pente moyenne = dénivelé net / longueur, avec altitudes interpolées aux bornes
    alt_bornes = np.interp(bornes, distance, altitude)
    longueur = np.diff(bornes)

    seg = pd.DataFrame(
        {
            "segment": np.arange(nb),
            "debut_m": bornes[:-1],
            "fin_m": bornes[1:],
            "longueur_m": longueur,
            "alt_debut_m": alt_bornes[:-1],
            "alt_fin_m": alt_bornes[1:],
            "d_plus_m": d_plus,
            "d_moins_m": d_moins,
            "pente": np.diff(alt_bornes) / longueur,
        }
    )
    seg["d_plus_cumul_m"] = seg["d_plus_m"].cumsum()
    return seg
