"""Profil athlète : les réglages personnels, sauvegardables en JSON et rechargeables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace

from trailnutri.config import Config


class ProfilError(ValueError):
    """Fichier de profil ou saisie de produits invalide."""


@dataclass(frozen=True)
class ProfilAthlete:
    derive_fatigue: float          # 0.10 = +10 % en fin de course
    facteur_descente_min: float    # plancher de descente (calibrable)
    glucides_g_h: int
    intervalle_min: int
    delai_premiere_prise_min: int
    produits: dict[str, float]     # produits utilisés (catalogue + perso) : nom -> g de glucides
    temperature_c: int

    @classmethod
    def depuis_config(cls, cfg: Config) -> ProfilAthlete:
        return cls(
            derive_fatigue=cfg.pacing.derive_fatigue,
            facteur_descente_min=cfg.pacing.facteur_descente_min,
            glucides_g_h=int(cfg.nutrition.glucides_g_h),
            intervalle_min=int(cfg.nutrition.intervalle_min),
            delai_premiere_prise_min=int(cfg.nutrition.delai_premiere_prise_min),
            produits=dict(cfg.nutrition.produits),
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
                produits=dict(self.produits),
            ),
            hydratation=replace(cfg.hydratation, temperature_c=self.temperature_c),
        )

    def vers_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def depuis_json(cls, texte: str, catalogue: dict[str, float] | None = None) -> ProfilAthlete:
        """Lit un profil ; refuse les clés manquantes/inconnues et les types incorrects.

        Ancien format (produits = liste de noms) : les valeurs sont prises dans `catalogue`.
        """
        try:
            brut = json.loads(texte)
        except json.JSONDecodeError as erreur:
            raise ProfilError(f"Profil illisible (JSON invalide) : {erreur}") from erreur
        if not isinstance(brut, dict):
            raise ProfilError("Profil invalide : un objet JSON est attendu.")

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
            if nom == "produits":
                valeur = _lire_produits(valeur, catalogue or {})
            elif not isinstance(valeur, (int, float)) or isinstance(valeur, bool):
                raise ProfilError(f"'{nom}' doit être un nombre.")
            else:
                valeur = int(valeur) if type_ == "int" else float(valeur)
            valeurs[nom] = valeur
        return cls(**valeurs)


def produits_depuis_saisie(
    lignes: Iterable[tuple[object, object]], glucides_max_g: float
) -> dict[str, float]:
    """Transforme les lignes saisies (nom, glucides) en produits, en les validant.

    Lignes entièrement vides ignorées ; nom vide, glucides manquants ou hors ]0, max],
    doublon de nom -> ProfilError.
    """
    produits: dict[str, float] = {}
    for nom_brut, glucides in lignes:
        nom = str(nom_brut).strip() if _rempli(nom_brut) else ""
        if not nom and not _rempli(glucides):
            continue  # ligne ajoutée puis laissée vide
        if not nom:
            raise ProfilError("Un produit perso n'a pas de nom.")
        if not _rempli(glucides):
            raise ProfilError(f"Glucides manquants pour « {nom} ».")
        if not 0 < float(glucides) <= glucides_max_g:
            raise ProfilError(
                f"« {nom} » : glucides entre 1 et {glucides_max_g:g} g par unité attendus."
            )
        if nom.casefold() in (n.casefold() for n in produits):
            raise ProfilError(f"Le produit « {nom} » apparaît deux fois.")
        produits[nom] = float(glucides)
    return produits


def _lire_produits(valeur: object, catalogue: dict[str, float]) -> dict[str, float]:
    """Produits du JSON : {nom: g} (format actuel) ou [noms] (ancien format)."""
    if isinstance(valeur, list) and all(isinstance(v, str) for v in valeur):
        return {nom: catalogue[nom] for nom in valeur if nom in catalogue}
    if isinstance(valeur, dict) and all(
        isinstance(k, str) and isinstance(g, (int, float)) and not isinstance(g, bool) and g > 0
        for k, g in valeur.items()
    ):
        return {k: float(g) for k, g in valeur.items()}
    raise ProfilError("'produits' doit associer chaque nom de produit à ses glucides (g > 0).")


def _rempli(valeur: object) -> bool:
    """Faux pour None, NaN (cellule vide du tableau) et texte vide."""
    if valeur is None:
        return False
    if isinstance(valeur, float) and valeur != valeur:  # NaN
        return False
    return str(valeur).strip() != ""
