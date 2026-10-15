"""Profil athlète : les réglages personnels, sauvegardables en JSON et rechargeables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace

from trailnutri.config import Config

# Clés de profils enregistrés par d'anciennes versions, ignorées à la lecture
CLES_OBSOLETES = ("produits",)


class ProfilError(ValueError):
    """Fichier de profil invalide."""


@dataclass(frozen=True)
class ProfilAthlete:
    derive_fatigue: float          # 0.10 = +10 % en fin de course
    facteur_descente_min: float    # plancher de descente (calibrable)
    glucides_g_h: int
    intervalle_min: int
    delai_premiere_prise_min: int
    temperature_c: int

    @classmethod
    def depuis_config(cls, cfg: Config) -> ProfilAthlete:
        return cls(
            derive_fatigue=cfg.pacing.derive_fatigue,
            facteur_descente_min=cfg.pacing.facteur_descente_min,
            glucides_g_h=int(cfg.nutrition.glucides_g_h),
            intervalle_min=int(cfg.nutrition.intervalle_min),
            delai_premiere_prise_min=int(cfg.nutrition.delai_premiere_prise_min),
            temperature_c=int(cfg.hydratation.temperature_c),
        )

    def appliquer(self, cfg: Config) -> Config:
        """Config surchargée par le profil (la config d'origine n'est pas modifiée)."""
        return replace(
            cfg,
            pacing=replace(
                cfg.pacing,
                derive_fatigue=self.derive_fatigue,
                facteur_descente_min=self.facteur_descente_min,
            ),
            nutrition=replace(
                cfg.nutrition,
                glucides_g_h=self.glucides_g_h,
                intervalle_min=self.intervalle_min,
                delai_premiere_prise_min=self.delai_premiere_prise_min,
            ),
            hydratation=replace(cfg.hydratation, temperature_c=self.temperature_c),
        )

    def vers_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def depuis_json(cls, texte: str) -> ProfilAthlete:
        """Lit un profil ; refuse les clés manquantes/inconnues et les types incorrects.

        Les clés d'anciennes versions (CLES_OBSOLETES) sont ignorées.
        """
        try:
            brut = json.loads(texte)
        except json.JSONDecodeError as erreur:
            raise ProfilError(f"Profil illisible (JSON invalide) : {erreur}") from erreur
        if not isinstance(brut, dict):
            raise ProfilError("Profil invalide : un objet JSON est attendu.")
        for cle in CLES_OBSOLETES:
            brut.pop(cle, None)

        attendus = {f.name: f.type for f in fields(cls)}
        manquants = attendus.keys() - brut.keys()
        inconnus = brut.keys() - attendus.keys()
        if manquants or inconnus:
            raise ProfilError(
                f"Profil invalide : clés manquantes {sorted(manquants)}, "
                f"inconnues {sorted(inconnus)}."
            )

        valeurs = {}
        for nom, type_ in attendus.items():
            valeur = brut[nom]
            if not isinstance(valeur, (int, float)) or isinstance(valeur, bool):
                raise ProfilError(f"'{nom}' doit être un nombre.")
            else:
                valeur = int(valeur) if type_ == "int" else float(valeur)
            valeurs[nom] = valeur
        return cls(**valeurs)
