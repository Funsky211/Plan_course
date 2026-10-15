from dataclasses import replace

import pytest
import yaml

from trailnutri.config import CHEMIN_CONFIG_DEFAUT, ConfigError, charger_config, valider


def test_config_par_defaut_valide(cfg):
    valider(cfg)  # ne lève rien


@pytest.mark.parametrize(
    "section, modif, message",
    [
        ("pacing", {"facteur_descente_min": 1.5}, "facteur_descente_min"),
        ("pacing", {"pente_min": 0.1}, "pente_min"),
        ("difficulte", {"seuil_zone_dure": 2}, "seuil_zone_dure"),
        ("difficulte", {"creux_debut_h": 5}, "creux_debut_h"),
        ("nutrition", {"glucides_g_h": 150}, "glucides_g_h"),
        ("nutrition", {"produits": {"Gel": -1}}, "produits"),
        ("hydratation", {"ml_h_min": 1000}, "ml_h_min"),
    ],
)
def test_valeurs_incoherentes(cfg, section, modif, message):
    mauvaise = replace(cfg, **{section: replace(getattr(cfg, section), **modif)})
    with pytest.raises(ConfigError, match=message):
        valider(mauvaise)


def test_poids_difficulte_incomplets(cfg):
    poids = dict(cfg.difficulte.poids)
    del poids["creux"]
    with pytest.raises(ConfigError, match="difficulte.poids"):
        valider(replace(cfg, difficulte=replace(cfg.difficulte, poids=poids)))


def test_toutes_les_erreurs_listees(cfg):
    mauvaise = replace(cfg, segments=replace(cfg.segments, longueur_m=0),
                       decalage=replace(cfg.decalage, pas_min=0))
    with pytest.raises(ConfigError) as erreur:
        valider(mauvaise)
    assert "segments.longueur_m" in str(erreur.value)
    assert "decalage" in str(erreur.value)


def test_cle_manquante_ou_inconnue(tmp_path):
    brut = yaml.safe_load(CHEMIN_CONFIG_DEFAUT.read_text(encoding="utf-8"))
    brut["pacing"]["vitesse_lumiere"] = 3e8
    chemin = tmp_path / "config.yaml"
    chemin.write_text(yaml.safe_dump(brut, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="manquante / inconnue"):
        charger_config(chemin)

    del brut["pacing"]
    chemin.write_text(yaml.safe_dump(brut, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError):
        charger_config(chemin)
