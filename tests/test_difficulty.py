from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trailnutri.difficulty import (
    _niveaux_par_zone,
    calculer_difficulte,
    distance_montee_continue,
    lisser,
    niveau,
    niveaux_hysteresis,
    zones,
)

FACTEURS = ["montee", "fatigue", "position", "descente", "enchainement", "creux"]


def segments_test(pentes, duree_seg_s: float = 60, longueur: float = 100) -> pd.DataFrame:
    """Segments avec temps : chaque segment dure duree_seg_s."""
    pentes = np.asarray(pentes, dtype=float)
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


def test_facteurs_et_score_entre_0_et_1(cfg):
    pentes = [0.0] * 10 + [0.12] * 10 + [-0.20] * 6 + [0.05] * 10
    seg = calculer_difficulte(segments_test(pentes, duree_seg_s=600), cfg.difficulte)
    for f in FACTEURS:
        assert seg[f"f_{f}"].between(0, 1).all(), f
    assert seg["difficulte"].between(0, 1).all()
    assert seg["difficulte_brute"].between(0, 1).all()


def test_distance_montee_continue():
    pente = np.array([0.0, 0.05, 0.10, 0.02, 0.06, 0.07])
    assert list(distance_montee_continue(pente, np.full(6, 100.0), 0.03)) == [
        0, 100, 200, 0, 100, 200
    ]


def test_montee_progressive_avec_la_longueur(cfg):
    # Pente constante de 10 % : le facteur montée croît au fil des 500 premiers mètres
    seg = calculer_difficulte(segments_test([0.10] * 10), cfg.difficulte)
    f = seg["f_montee"].to_numpy()
    assert (np.diff(f[:5]) > 0).all()
    assert f[4] == pytest.approx(f[9])  # plein à partir de 500 m
    assert seg["montee_dure"].iloc[-1]


def test_montee_continue_en_pente(cfg):
    # Montées longues : le facteur croît continûment avec la pente (plus de marche d'escalier)
    pentes = np.linspace(0.0, 0.20, 41)
    facteurs = np.array([
        calculer_difficulte(segments_test([p] * 8), cfg.difficulte)["f_montee"].iloc[-1]
        for p in pentes
    ])
    au_dessus = pentes > cfg.difficulte.montee_plancher
    assert (facteurs[~au_dessus] == 0).all()  # sous le plancher : pas de montée continue
    assert (np.diff(facteurs[au_dessus]) > 0).all()
    assert facteurs[-1] > 0.95


def test_descente_technique_continue(cfg):
    seg = calculer_difficulte(segments_test(np.linspace(0, -0.30, 31)), cfg.difficulte)
    f = seg["f_descente"].to_numpy()
    assert (np.diff(f) > 0).all()
    i = int(np.argmin(np.abs(seg["pente"] + 0.12)))
    assert f[i] == pytest.approx(0.5, abs=0.01)  # centre à -12 %


def test_creux_physiologique_en_cloche(cfg):
    # 60 segments de 5 min = 5 h : maximum vers 3h15, symétrique
    seg = calculer_difficulte(segments_test([0.0] * 60, duree_seg_s=300), cfg.difficulte)
    heures = (seg["temps_debut_s"] + seg["temps_fin_s"]) / 2 / 3600
    pic = heures.iloc[seg["f_creux"].idxmax()]
    assert pic == pytest.approx(3.25, abs=0.05)
    assert seg["f_creux"].iloc[0] < 0.01


def test_fatigue_et_position_progressives(cfg):
    seg = calculer_difficulte(segments_test([0.0] * 60, duree_seg_s=300), cfg.difficulte)
    assert (np.diff(seg["f_fatigue"]) > 0).all()  # 5 h < 6 h : jamais saturée
    assert seg["f_position"].is_monotonic_increasing
    assert seg["f_position"].iloc[0] == 0 and seg["f_position"].iloc[-1] > 0.95


def test_enchainement_decroit_avec_la_recuperation(cfg):
    montee = [0.12] * 8
    valeurs = []
    for recup in (2, 5, 10, 20):  # x 100 m de plat entre deux montées
        seg = calculer_difficulte(segments_test(montee + [0.0] * recup + montee), cfg.difficulte)
        valeurs.append(seg["f_enchainement"].iloc[-1])
    assert valeurs == sorted(valeurs, reverse=True)
    assert valeurs[0] > 0 and valeurs[-1] == 0  # 2 km de récup > 1.5 km : plus d'enchaînement
    seg = calculer_difficulte(segments_test(montee + [0.0] * 2 + montee), cfg.difficulte)
    assert seg["f_enchainement"].iloc[0] == 0  # la première montée n'enchaîne rien


def test_score_continu_entre_voisins(cfg):
    # Deux segments voisins aux pentes proches ont des scores proches
    pentes = np.linspace(0.02, 0.14, 60)
    seg = calculer_difficulte(segments_test(pentes, duree_seg_s=120), cfg.difficulte)
    assert np.abs(np.diff(seg["difficulte"])).max() < 0.03
    assert np.abs(np.diff(seg["difficulte_brute"])).max() < 0.05


def test_lissage_supprime_les_sauts(cfg):
    # Un pic isolé (un segment très raide) est atténué par le lissage
    pentes = [0.0] * 20 + [0.25] + [0.0] * 20
    brut = replace(cfg.difficulte, lissage_fenetre_m=0)
    seg_brut = calculer_difficulte(segments_test(pentes), brut)
    seg_lisse = calculer_difficulte(segments_test(pentes), cfg.difficulte)
    assert np.abs(np.diff(seg_lisse["difficulte"])).max() < np.abs(
        np.diff(seg_brut["difficulte"])
    ).max()
    assert seg_brut["difficulte"].equals(seg_brut["difficulte_brute"])


def test_lisser_fenetre():
    score = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0])
    lisse = lisser(score, np.full(5, 100.0), 300)  # 3 segments
    assert list(lisse) == pytest.approx([0, 1 / 3, 1 / 3, 1 / 3, 0])
    assert lisser(score, np.full(5, 100.0), 0).equals(score)


def test_poids_normalises(cfg):
    # Multiplier tous les poids par 2 ne change pas le score (renormalisation)
    seg = segments_test([0.0] * 5 + [0.15] * 8 + [-0.20] * 5, duree_seg_s=900)
    double = replace(cfg.difficulte, poids={f: 2 * w for f, w in cfg.difficulte.poids.items()})
    a = calculer_difficulte(seg, cfg.difficulte)["difficulte"]
    b = calculer_difficulte(seg, double)["difficulte"]
    assert a.to_numpy() == pytest.approx(b.to_numpy())


# --- Niveaux et zones ----------------------------------------------------------------------

def test_niveau_aux_seuils(cfg):
    d = cfg.difficulte  # modéré à partir de 0.22, dur à partir de 0.38
    scores = [0.0, 0.2199, 0.22, 0.3799, 0.38, 1.0]
    assert list(niveau(scores, d)) == ["facile", "facile", "modéré", "modéré", "dur", "dur"]


def avec_scores(scores, longueur: float = 100) -> pd.DataFrame:
    """Segments dont le score lissé est imposé (pour tester niveaux et zones seuls)."""
    seg = segments_test(np.zeros(len(scores)), longueur=longueur)
    seg["difficulte"] = scores
    return seg


def test_zones_fusion_et_statistiques(cfg):
    scores = [0.1] * 5 + [0.3] * 5 + [0.5] * 4 + [0.1] * 6
    seg = avec_scores(scores)
    seg["niveau"] = _niveaux_par_zone(seg, cfg.difficulte)
    z = zones(seg)
    assert list(z["niveau"]) == ["facile", "modéré", "dur", "facile"]
    assert list(z["km_debut"]) == pytest.approx([0, 0.5, 1.0, 1.4])
    assert list(z["km_fin"]) == pytest.approx([0.5, 1.0, 1.4, 2.0])
    assert z.loc[2, "score_moyen"] == pytest.approx(0.5)
    assert (z["score_max"] >= z["score_moyen"] - 1e-12).all()


def test_pic_isole_absorbe(cfg):
    # Un seul segment "dur" (100 m) au milieu du facile : pas de zone qui clignote
    scores = [0.1] * 10 + [0.6] + [0.1] * 10
    seg = avec_scores(scores)
    assert set(_niveaux_par_zone(seg, cfg.difficulte)) == {"facile"}


def test_zone_courte_rejoint_la_voisine_la_plus_proche(cfg):
    # 200 m de "modéré" (0.36) entre du facile (0.1) et du dur (0.5) : rejoint le dur
    scores = [0.1] * 5 + [0.36] * 2 + [0.5] * 5
    seg = avec_scores(scores)
    niveaux = _niveaux_par_zone(seg, cfg.difficulte)
    assert list(niveaux[5:7]) == ["dur", "dur"]


@pytest.mark.parametrize("graine", range(5))
def test_aucune_zone_trop_courte(cfg, graine):
    # Scores bruités : toutes les zones font au moins la longueur minimale, et elles se suivent
    rng = np.random.default_rng(graine)
    seg = avec_scores(np.clip(rng.normal(0.3, 0.15, 120), 0, 1))
    seg["niveau"] = _niveaux_par_zone(seg, cfg.difficulte)
    z = zones(seg)
    longueurs_m = (z["km_fin"] - z["km_debut"]) * 1000
    assert (longueurs_m >= cfg.difficulte.zone_longueur_min_m - 1e-6).all() or len(z) == 1
    assert z["km_debut"].iloc[0] == 0 and z["km_fin"].iloc[-1] == pytest.approx(12.0)
    assert (z["km_debut"].iloc[1:].to_numpy() == pytest.approx(z["km_fin"].iloc[:-1].to_numpy()))
    assert (z["niveau"].iloc[1:].to_numpy() != z["niveau"].iloc[:-1].to_numpy()).all()


def test_course_plus_courte_que_le_minimum(cfg):
    seg = avec_scores([0.1, 0.5])
    assert len(set(_niveaux_par_zone(seg, cfg.difficulte))) == 1


def test_niveau_dans_calculer_difficulte(cfg):
    seg = calculer_difficulte(segments_test([0.0] * 10 + [0.15] * 20), cfg.difficulte)
    assert set(seg["niveau"]) <= {"facile", "modéré", "dur"}
    assert len(zones(seg)) >= 1


def test_hysteresis_evite_l_alternance(cfg):
    # Score qui oscille autour du seuil modéré (0.22) : un seul niveau, pas d'alternance
    scores = np.array([0.21, 0.23] * 20)
    assert len(set(niveaux_hysteresis(scores, cfg.difficulte))) == 1
    # Sans hystérésis, les niveaux alternent bien à chaque segment
    sans = replace(cfg.difficulte, hysteresis=0)
    assert len(set(niveaux_hysteresis(scores, sans))) == 2


def test_hysteresis_franchissement_net(cfg):
    # Un franchissement net (au-delà de la marge) change bien le niveau, dans les deux sens
    scores = np.array([0.1] * 5 + [0.30] * 5 + [0.45] * 5 + [0.30] * 5 + [0.1] * 5)
    niveaux = list(niveaux_hysteresis(scores, cfg.difficulte))
    assert niveaux[::5] == ["facile", "modéré", "dur", "modéré", "facile"]


def test_sans_hysteresis_egal_niveau_brut(cfg):
    sans = replace(cfg.difficulte, hysteresis=0)
    scores = np.random.default_rng(1).uniform(0, 1, 200)
    assert list(niveaux_hysteresis(scores, sans)) == list(niveau(scores, sans))
