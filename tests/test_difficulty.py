from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trailnutri.difficulty import blocs_montee, calculer_difficulte, zones_dures


def segments_test(pentes: list[float], duree_seg_s: float = 60, longueur: float = 100):
    """Segments avec temps : chaque segment dure duree_seg_s."""
    n = len(pentes)
    debut = np.arange(n) * longueur
    fin_s = (np.arange(n) + 1) * duree_seg_s
    return pd.DataFrame(
        {
            "debut_m": debut,
            "fin_m": debut + longueur,
            "longueur_m": longueur,
            "pente": pentes,
            "temps_debut_s": fin_s - duree_seg_s,
            "temps_fin_s": fin_s,
        }
    )


def test_bloc_trop_court_ignore(cfg):
    # 400 m à 10 % : en dessous des 500 m requis
    seg = segments_test([0.0] * 5 + [0.10] * 4 + [0.0] * 5)
    assert blocs_montee(seg, cfg.difficulte) == []


def test_bloc_long_detecte(cfg):
    seg = segments_test([0.0] * 5 + [0.10] * 6 + [0.0] * 5)
    assert blocs_montee(seg, cfg.difficulte) == [(5, 10)]


def test_replat_court_ne_coupe_pas(cfg):
    # 300 m + replat de 100 m + 300 m : une seule montée de 700 m
    seg = segments_test([0.0] * 3 + [0.10] * 3 + [0.02] + [0.10] * 3 + [0.0] * 3)
    assert blocs_montee(seg, cfg.difficulte) == [(3, 9)]


def test_facteurs_bornes_et_score(cfg):
    pentes = [0.0] * 10 + [0.12] * 8 + [-0.20] * 6 + [0.0] * 10
    seg = calculer_difficulte(segments_test(pentes), cfg.difficulte)
    for f in ["montee", "fatigue", "position", "descente", "enchainement", "creux"]:
        assert seg[f"f_{f}"].between(0, 1).all()
    assert seg["difficulte"].between(0, 1).all()
    assert seg.loc[10, "f_montee"] == pytest.approx(0.12 / 0.25)
    assert seg.loc[20, "descente_technique"]
    assert seg.loc[20, "f_descente"] == pytest.approx(0.20 / 0.30)
    assert not seg.loc[0, "montee_dure"]


def test_position_et_fatigue(cfg):
    seg = calculer_difficulte(segments_test([0.0] * 30), cfg.difficulte)
    assert seg["f_fatigue"].is_monotonic_increasing
    assert seg["f_position"].iloc[0] == 0 and seg["f_position"].iloc[-1] == 1


def test_creux_physiologique(cfg):
    # 60 segments de 5 min = 5 h : le creux couvre 2h30 - 4h
    seg = calculer_difficulte(segments_test([0.0] * 60, duree_seg_s=300), cfg.difficulte)
    heures = (seg["temps_debut_s"] + seg["temps_fin_s"]) / 2 / 3600
    assert (seg.loc[heures.between(2.5, 4.0), "f_creux"] == 1).all()
    assert (seg.loc[heures < 2.5, "f_creux"] == 0).all()


def test_enchainement(cfg):
    montee = [0.12] * 6
    proches = segments_test(montee + [0.0] * 3 + montee)      # 300 m de récup
    eloignees = segments_test(montee + [0.0] * 20 + montee)   # 2 km de récup
    s_proches = calculer_difficulte(proches, cfg.difficulte)
    s_eloignees = calculer_difficulte(eloignees, cfg.difficulte)
    assert s_proches["f_enchainement"].iloc[-1] == pytest.approx(1 - 300 / 1500)
    assert s_eloignees["f_enchainement"].max() == 0
    assert s_proches["f_enchainement"].iloc[0] == 0  # la première montée n'enchaîne rien


def test_poids_normalises(cfg):
    # Multiplier tous les poids par 2 ne change pas le score (renormalisation)
    seg = segments_test([0.0] * 5 + [0.15] * 8 + [-0.20] * 5, duree_seg_s=900)
    double = replace(cfg.difficulte, poids={f: 2 * w for f, w in cfg.difficulte.poids.items()})
    a = calculer_difficulte(seg, cfg.difficulte)["difficulte"]
    b = calculer_difficulte(seg, double)["difficulte"]
    assert a.to_numpy() == pytest.approx(b.to_numpy())


def test_zones_dures(cfg):
    # Course courte (30 min) : ni creux ni grosse fatigue, seule la montée raide ressort
    pentes = [0.0] * 10 + [0.25] * 10 + [0.0] * 10
    seg = calculer_difficulte(segments_test(pentes, duree_seg_s=60), cfg.difficulte)
    assert zones_dures(seg) == [(pytest.approx(1.0), pytest.approx(2.0))]
