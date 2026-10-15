"""Petits utilitaires de formatage / lecture de durées et d'heures."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta


def formater_hmm(secondes: float) -> str:
    """Durée -> 'h:mm' (ex. 5400 -> '1:30')."""
    minutes = int(round(secondes / 60))
    return f"{minutes // 60}:{minutes % 60:02d}"


def formater_mmss(secondes: float) -> str:
    """Durée -> 'm:ss' (ex. 330 -> '5:30'), pour les allures."""
    secondes = int(round(secondes))
    return f"{secondes // 60}:{secondes % 60:02d}"


def formater_heure(secondes: float, heure_depart: time | None) -> str:
    """Temps écoulé 'h:mm', ou heure réelle 'HH:MM' si un départ est donné."""
    if heure_depart is None:
        return formater_hmm(secondes)
    instant = datetime.combine(datetime.today(), heure_depart) + timedelta(seconds=secondes)
    return instant.strftime("%H:%M")


def lire_hmm(texte: str) -> float:
    """'2:30', '2h30', '2h' -> secondes. Lève ValueError si le format est inconnu."""
    m = re.fullmatch(r"\s*(\d{1,2})\s*[h:]\s*(\d{2})?\s*", texte.lower())
    if not m:
        raise ValueError(f"Durée/heure illisible : '{texte}' (attendu h:mm ou hhmm, ex. 2:30)")
    heures, minutes = int(m.group(1)), int(m.group(2) or 0)
    if minutes >= 60:
        raise ValueError(f"Minutes invalides dans '{texte}'")
    return heures * 3600 + minutes * 60


def secondes_depuis_depart(heure: time, heure_depart: time) -> float:
    """Heure réelle -> temps écoulé depuis le départ (la course peut passer minuit)."""
    delta = datetime.combine(datetime.today(), heure) - datetime.combine(
        datetime.today(), heure_depart
    )
    return delta.total_seconds() % 86_400
