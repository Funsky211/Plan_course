from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd
import pytest

from trailnutri.nutrition import NutritionError, exporter_csv, plan_glucides
from trailnutri.waypoints import Ravito


def segments_temps(duree_totale_s: float, nb: int = 10, longueur: float = 1_000) -> pd.DataFrame:
    """Course plate fictive : nb segments de même durée, 10 m de D+ par segment."""
    fin = [longueur * (k + 1) for k in range(nb)]
    return pd.DataFrame(
        {
            "fin_m": fin,
            "temps_fin_s": [duree_totale_s * (k + 1) / nb for k in range(nb)],
            "d_plus_cumul_m": [10.0 * (k + 1) for k in range(nb)],
        }
    )


def test_horaires_et_quantites(cfg):
    # 2 h, une prise toutes les 25 min à 60 g/h -> 25, 50, 75, 100 min ; 25 g par prise
    plan = plan_glucides(segments_temps(2 * 3600), cfg.nutrition)
    assert list(plan["temps_s"] / 60) == [25, 50, 75, 100]
    assert (plan["glucides_g"] == 25).all()
    assert list(plan["heure"]) == ["0:25", "0:50", "1:15", "1:40"]
    assert (plan["raison"] == "intervalle fixe").all()


def test_position_interpolee(cfg):
    # Allure constante : 10 km en 100 min -> à 25 min on est au km 2.5, D+ 25 m
    plan = plan_glucides(segments_temps(100 * 60), cfg.nutrition)
    assert plan["km"].iloc[0] == pytest.approx(2.5)
    assert plan["d_plus_cumul_m"].iloc[0] == pytest.approx(25)


def test_delai_premiere_prise(cfg):
    nutri = replace(cfg.nutrition, intervalle_min=10, delai_premiere_prise_min=15)
    plan = plan_glucides(segments_temps(3600), nutri)
    # 10 min est dans le délai : première prise à 20 min
    assert plan["temps_s"].iloc[0] == 20 * 60
    assert plan["glucides_g"].iloc[0] == pytest.approx(10)


def test_pas_de_prise_a_l_arrivee(cfg):
    # Course de 50 min pile : la prise de 50 min tomberait sur la ligne, on l'exclut
    plan = plan_glucides(segments_temps(50 * 60), cfg.nutrition)
    assert list(plan["temps_s"] / 60) == [25]


def test_course_courte_sans_prise(cfg):
    plan = plan_glucides(segments_temps(20 * 60), cfg.nutrition)
    assert plan.empty


def test_heure_reelle(cfg):
    plan = plan_glucides(segments_temps(3600), cfg.nutrition, heure_depart=time(7, 50))
    assert list(plan["heure"]) == ["08:15", "08:40"]


@pytest.mark.parametrize(
    "modif",
    [{"glucides_g_h": 0}, {"glucides_g_h": 120}, {"intervalle_min": 0}, {"delai_premiere_prise_min": -5}],
)
def test_parametres_invalides(cfg, modif):
    with pytest.raises(NutritionError):
        plan_glucides(segments_temps(3600), replace(cfg.nutrition, **modif))


def test_export_csv(cfg):
    csv = exporter_csv(plan_glucides(segments_temps(100 * 60), cfg.nutrition))
    lignes = csv.strip().splitlines()
    assert lignes[0] == "heure;km;d_plus_cumul_m;glucides_g;raison"
    assert lignes[1] == "0:25;2,5;25,0;25,0;intervalle fixe"


# --- Décalage des prises -------------------------------------------------------------------

def segments_terrain(montee=(), descente=(), nb_min: int = 120) -> pd.DataFrame:
    """Un segment de 200 m par minute. `montee` / `descente` : minutes (début, fin exclue)."""
    minutes = np.arange(nb_min)
    montee_dure = (minutes >= montee[0]) & (minutes < montee[1]) if montee else minutes < 0
    descente_tech = (minutes >= descente[0]) & (minutes < descente[1]) if descente else minutes < 0
    pente = np.where(montee_dure, 0.15, np.where(descente_tech, -0.20, 0.0))
    return pd.DataFrame(
        {
            "debut_m": minutes * 200.0,
            "fin_m": (minutes + 1) * 200.0,
            "pente": pente,
            "temps_debut_s": minutes * 60.0,
            "temps_fin_s": (minutes + 1) * 60.0,
            "d_plus_cumul_m": np.cumsum(np.clip(pente, 0, None) * 200),
            "montee_dure": montee_dure,
            "descente_technique": descente_tech,
        }
    )


def premiere_prise(cfg, segments, ravitos=None, **modif_nutrition):
    nutri = replace(cfg.nutrition, **modif_nutrition)
    plan = plan_glucides(segments, nutri, decalage=cfg.decalage, ravitos=ravitos)
    return plan.iloc[0]


def test_prise_avancee_avant_la_montee(cfg):
    # Montée dure de 20 à 35 min : la prise de 25 min recule à 19 min, juste avant
    prise = premiere_prise(cfg, segments_terrain(montee=(20, 35)))
    assert prise["temps_s"] == 19 * 60
    assert prise["decalage_min"] == -6
    assert prise["raison"].startswith("avant la montée du km 4.0")


def test_prise_deja_bien_placee(cfg):
    # La montée commence à 28 min : 25 min est déjà "juste avant", on ne bouge pas
    prise = premiere_prise(cfg, segments_terrain(montee=(28, 40)))
    assert prise["temps_s"] == 25 * 60
    assert prise["raison"] == "avant la montée du km 5.6"


def test_calage_sur_ravito(cfg):
    # Ravito au km 5.6 = 28 min (200 m/min)
    prise = premiere_prise(cfg, segments_terrain(), ravitos=[Ravito(5.6, "Col")])
    assert prise["temps_s"] == pytest.approx(28 * 60)
    assert "au ravito « Col »" in prise["raison"]


def test_evite_descente_technique(cfg):
    # Descente technique de 22 à 30 min : on sort de la descente au plus près (21 min)
    prise = premiere_prise(cfg, segments_terrain(descente=(22, 30)))
    assert prise["temps_s"] == 21 * 60
    assert "descente" not in prise["raison"]


def test_pas_de_creneau_signale(cfg):
    # Descente technique sur toute la fenêtre : la prise reste, avec un avertissement
    prise = premiere_prise(cfg, segments_terrain(descente=(10, 60)))
    assert "en descente technique, pas de meilleur créneau" in prise["raison"]


def test_ecart_minimal_entre_prises(cfg):
    # Ravito à 21 min attire la 1re prise (15 -> 21) ; la 2e (30) doit rester à >= 12 min
    nutri = replace(cfg.nutrition, intervalle_min=15)
    plan = plan_glucides(segments_terrain(), nutri, decalage=cfg.decalage,
                         ravitos=[Ravito(4.2, "R")])
    assert plan["temps_s"].iloc[0] == pytest.approx(21 * 60)
    assert (np.diff(plan["temps_s"]) >= 12 * 60 - 1e-6).all()


def test_fenetre_respectee(cfg):
    plan = plan_glucides(segments_terrain(montee=(20, 35), descente=(50, 70)), cfg.nutrition,
                         decalage=cfg.decalage)
    assert plan["decalage_min"].abs().max() <= cfg.decalage.fenetre_min


def test_decalage_exige_difficulte(cfg):
    with pytest.raises(ValueError, match="calculer_difficulte"):
        plan_glucides(segments_temps(3600), cfg.nutrition, decalage=cfg.decalage)
