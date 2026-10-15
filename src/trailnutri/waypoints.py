"""Points d'intérêt (POI) : lecture des <wpt> du GPX, projection sur la trace, interprétation.

Les POI sont classés par mots-clés (config.yaml) : ravito, eau, barriere, sommet, col...
Les ravitos et barrières peuvent aussi être saisis à la main (config ou interface).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import time

import gpxpy.gpx
import numpy as np
import pandas as pd

from trailnutri.config import CourseConfig, WaypointsConfig
from trailnutri.gpx_loader import distance_haversine
from trailnutri.temps import lire_hmm, secondes_depuis_depart

# "14h30", "14h", "14:30" dans le nom ou la description d'une barrière
MOTIF_HEURE = re.compile(r"\b(\d{1,2})\s*[h:]\s*(\d{2})?\b")


@dataclass(frozen=True)
class Ravito:
    km: float
    nom: str
    type: str = "ravito"  # "ravito" (solide + liquide) ou "eau" (liquide seulement)


@dataclass(frozen=True)
class Barriere:
    km: float
    limite_s: float  # temps écoulé maximum depuis le départ
    nom: str


def extraire_pois(gpx: gpxpy.gpx.GPX, trace: pd.DataFrame, cfg: WaypointsConfig) -> pd.DataFrame:
    """Une ligne par passage d'un POI sur la trace (un aller-retour donne deux lignes).

    Colonnes : nom, categories (tuple), km (NaN si trop loin de la trace), ecart_trace_m,
    heure_limite ('HH:MM' ou ''), description.
    """
    lignes = []
    for wpt in gpx.waypoints:
        texte = " ".join(
            filter(None, [wpt.name, wpt.description, wpt.comment, wpt.symbol, wpt.type])
        )
        categories = classer(texte, cfg)
        heure = lire_heure_limite(texte) if "barriere" in categories else ""
        base = {
            "nom": wpt.name or "(sans nom)",
            "categories": categories,
            "heure_limite": heure,
            "description": wpt.description or "",
        }
        passages = _passages_sur_trace(wpt.latitude, wpt.longitude, trace, cfg)
        if not passages:
            ecart = distance_haversine(wpt.latitude, wpt.longitude, trace["lat"], trace["lon"])
            lignes.append({**base, "km": np.nan, "ecart_trace_m": float(np.min(ecart))})
        for km, ecart in passages:
            lignes.append({**base, "km": km, "ecart_trace_m": ecart})

    colonnes = ["nom", "categories", "km", "ecart_trace_m", "heure_limite", "description"]
    pois = pd.DataFrame(lignes, columns=colonnes)
    return pois.sort_values("km", na_position="last").reset_index(drop=True)


def classer(texte: str, cfg: WaypointsConfig) -> tuple[str, ...]:
    """Toutes les catégories dont un mot-clé apparaît (mot entier, sans casse ni accents)."""
    texte = _normaliser(texte)
    trouvees = []
    for categorie, mots in cfg.mots_cles.items():
        for mot in mots:
            if re.search(rf"\b{re.escape(_normaliser(mot))}\b", texte):
                trouvees.append(categorie)
                break
    return tuple(trouvees)


def lire_heure_limite(texte: str) -> str:
    """Première heure 'HH:MM' trouvée dans le texte, '' sinon."""
    for m in MOTIF_HEURE.finditer(texte):
        heures, minutes = int(m.group(1)), int(m.group(2) or 0)
        if heures < 24 and minutes < 60:
            return f"{heures:02d}:{minutes:02d}"
    return ""


def ravitos_depuis_pois(pois: pd.DataFrame) -> list[Ravito]:
    """Les POI 'ravito' ou 'eau' situés sur la trace."""
    ravitos = []
    for _, poi in pois.dropna(subset=["km"]).iterrows():
        if "ravito" in poi["categories"]:
            ravitos.append(Ravito(poi["km"], poi["nom"], "ravito"))
        elif "eau" in poi["categories"]:
            ravitos.append(Ravito(poi["km"], poi["nom"], "eau"))
    return ravitos


def barrieres_depuis_pois(pois: pd.DataFrame, heure_depart: time | None) -> list[Barriere]:
    """Les POI 'barriere' ayant une heure lue. L'heure du GPX est une heure réelle :
    sans heure de départ, impossible de la convertir -> barrière ignorée."""
    if heure_depart is None:
        return []
    barrieres = []
    for _, poi in pois.dropna(subset=["km"]).iterrows():
        if "barriere" in poi["categories"] and poi["heure_limite"]:
            heure = time.fromisoformat(poi["heure_limite"])
            limite = secondes_depuis_depart(heure, heure_depart)
            barrieres.append(Barriere(poi["km"], limite, poi["nom"]))
    return barrieres


def ravitos_depuis_config(cfg: CourseConfig) -> list[Ravito]:
    return [Ravito(float(km), f"Ravito km {km}") for km in cfg.ravitos_km]


def barrieres_depuis_config(cfg: CourseConfig) -> list[Barriere]:
    return [
        Barriere(float(b["km"]), lire_hmm(str(b["limite"])), f"Barrière km {b['km']}")
        for b in cfg.barrieres
    ]


def parser_ravitos(texte: str) -> list[Ravito]:
    """'5.2, 10 ; 14,5' -> ravitos aux km 5.2, 10 et 14.5. Lève ValueError si illisible."""
    ravitos = []
    for morceau in re.split(r"[;\s]+|,\s+", texte.strip()):
        if morceau:
            km = float(morceau.replace(",", "."))
            ravitos.append(Ravito(km, f"Ravito km {km:g}"))
    return ravitos


def parser_barrieres(texte: str) -> list[Barriere]:
    """Une barrière par ligne : 'km ; h:mm' (temps écoulé), ex. '12.5 ; 2:30'."""
    barrieres = []
    for ligne in texte.strip().splitlines():
        if not ligne.strip():
            continue
        try:
            km_txt, limite_txt = ligne.split(";")
        except ValueError:
            raise ValueError(f"Ligne de barrière illisible : '{ligne}' (attendu 'km ; h:mm')")
        km = float(km_txt.strip().replace(",", "."))
        barrieres.append(Barriere(km, lire_hmm(limite_txt), f"Barrière km {km:g}"))
    return barrieres


def fusionner_ravitos(ravitos: list[Ravito], ecart_min_km: float) -> list[Ravito]:
    """Trie par km et retire les doublons (même ravito venant du GPX et de la saisie)."""
    resultat: list[Ravito] = []
    for r in sorted(ravitos, key=lambda r: r.km):
        if resultat and r.km - resultat[-1].km < ecart_min_km:
            # Doublon : un ravito complet l'emporte sur un simple point d'eau
            if r.type == "ravito" and resultat[-1].type == "eau":
                resultat[-1] = r
            continue
        resultat.append(r)
    return resultat


def _passages_sur_trace(
    lat: float, lon: float, trace: pd.DataFrame, cfg: WaypointsConfig
) -> list[tuple[float, float]]:
    """(km, écart en m) pour chaque passage de la trace à moins de distance_max_trace_m du POI."""
    ecart = distance_haversine(lat, lon, trace["lat"].to_numpy(), trace["lon"].to_numpy())
    proche = ecart <= cfg.distance_max_trace_m
    distance = trace["distance_m"].to_numpy()

    passages: list[tuple[float, float]] = []
    # Chaque plage continue de points proches = un passage ; on garde le point le plus proche
    for debut, fin in _plages(proche):
        i = debut + int(np.argmin(ecart[debut : fin + 1]))
        km, e = distance[i] / 1000, float(ecart[i])
        # Deux plages trop rapprochées = un seul passage (bruit GPS en bord de rayon)
        if passages and (km - passages[-1][0]) * 1000 < cfg.ecart_min_passages_m:
            if e < passages[-1][1]:
                passages[-1] = (km, e)
            continue
        passages.append((km, e))
    return passages


def _plages(masque: np.ndarray) -> list[tuple[int, int]]:
    """Indices (début, fin inclus) des suites de True consécutifs."""
    bords = np.diff(np.concatenate([[0], masque.astype(int), [0]]))
    debuts = np.flatnonzero(bords == 1)
    fins = np.flatnonzero(bords == -1) - 1
    return list(zip(debuts.tolist(), fins.tolist()))


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents, apostrophes typographiques unifiées."""
    texte = unicodedata.normalize("NFKD", texte.lower().replace("’", "'"))
    return "".join(c for c in texte if not unicodedata.combining(c))
