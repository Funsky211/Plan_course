from dataclasses import replace
from datetime import time

import pandas as pd
import pytest

from trailnutri.nutrition import NutritionError, exporter_csv, plan_glucides


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
