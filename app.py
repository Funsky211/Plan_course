"""Interface Streamlit. Lancer : python -m streamlit run app.py"""

from __future__ import annotations

import io
from dataclasses import asdict, replace
from datetime import time
from pathlib import Path

import pandas as pd
import streamlit as st

from trailnutri.calibration import CalibrationError, calibrer
from trailnutri.config import ConfigError, charger_config, valider
from trailnutri.gpx_loader import GpxError, construire_trace, denivele, lire_gpx
from trailnutri.nutrition import NutritionError, exporter_csv, total_a_prevoir
from trailnutri.pacing import PacingError, allure_plate, passages_par_km, temps_au_km
from trailnutri.pipeline import Resultat, planifier, rassembler_points
from trailnutri.plot import graphique_allure, profil_altimetrique
from trailnutri.profil import ProfilAthlete, ProfilError
from trailnutri.segments import decouper_segments
from trailnutri.temps import formater_hmm, formater_mmss
from trailnutri.waypoints import extraire_pois, parser_barrieres, parser_ravitos

GPX_EXEMPLE = Path(__file__).parent / "data" / "exemple.gpx"
CFG = charger_config()

# Bornes des curseurs du profil athlète : (min, max, pas). Le type du min fixe celui du curseur.
CURSEURS = {
    "derive_fatigue": (0.0, 0.30, 0.01),
    "facteur_descente_min": (0.5, 1.0, 0.05),
    "glucides_g_h": (20, int(CFG.nutrition.glucides_g_h_max), 5),
    "intervalle_min": (10, 45, 1),
    "delai_premiere_prise_min": (0, 30, 1),
    "temperature_c": (-10, 40, 1),
}

# Niveau affiché avec un symbole de couleur + son nom (jamais la couleur seule)
ETIQUETTES_NIVEAU = {"facile": "🟢 facile", "modéré": "🟠 modéré", "dur": "🔴 dur"}

st.set_page_config(page_title="trailnutri", layout="wide")


# --- État du profil athlète (curseurs) ----------------------------------------------------

def ajuster_au_curseur(nom: str, valeur: float) -> float:
    """Borne la valeur, l'aligne sur le pas du curseur et lui donne son type (int / float)."""
    mini, maxi, pas = CURSEURS[nom]
    valeur = min(max(valeur, mini), maxi)
    return type(mini)(round(round((valeur - mini) / pas) * pas + mini, 2))


def ecrire_profil(profil: ProfilAthlete) -> None:
    """Pousse un profil dans les curseurs, en respectant leurs bornes et types."""
    for nom, valeur in asdict(profil).items():
        st.session_state[nom] = ajuster_au_curseur(nom, valeur)


def lire_profil() -> ProfilAthlete:
    champs = ProfilAthlete.__dataclass_fields__
    return ProfilAthlete(**{nom: st.session_state[nom] for nom in champs})


def charger_profil_json() -> None:
    """Callback de l'import : exécuté avant le rendu, donc avant la création des curseurs."""
    fichier = st.session_state.get("fichier_profil")
    if fichier is None:
        return
    try:
        ecrire_profil(ProfilAthlete.depuis_json(fichier.getvalue().decode("utf-8")))
        st.session_state["message_profil"] = ("ok", "Profil chargé.")
    except (ProfilError, UnicodeDecodeError) as erreur:
        st.session_state["message_profil"] = ("erreur", str(erreur))


def appliquer_calibration(derive: float, descente: float) -> None:
    st.session_state["derive_fatigue"] = ajuster_au_curseur("derive_fatigue", derive)
    st.session_state["facteur_descente_min"] = ajuster_au_curseur("facteur_descente_min", descente)


if "glucides_g_h" not in st.session_state:
    ecrire_profil(ProfilAthlete.depuis_config(CFG))


# --- Lecture GPX (mise en cache : ne se refait pas à chaque mouvement de curseur) ----------

@st.cache_data(show_spinner="Lecture du GPX...")
def analyser_gpx(contenu: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    gpx = lire_gpx(io.StringIO(contenu.decode("utf-8-sig")))
    trace = construire_trace(gpx, CFG.gpx)
    return trace, extraire_pois(gpx, trace, CFG.waypoints)


# --- Barre latérale ------------------------------------------------------------------------

with st.sidebar:
    st.header("Course")
    fichier = st.file_uploader("Trace GPX", type=["gpx"])
    st.caption("Sans fichier : trace d'exemple (~15 km, ~760 m D+, avec ravitos et barrière).")
    col_h, col_m = st.columns(2)
    heures = col_h.number_input("Temps cible (h)", min_value=0, max_value=72, value=2)
    minutes = col_m.number_input("(min)", min_value=0, max_value=59, value=0)
    heure_depart = st.time_input("Heure de départ", value=time(8, 0))
    decaler = st.checkbox("Décaler les prises selon le terrain", value=True)

    st.header("Ravitos et barrières")
    utiliser_pois = st.checkbox("Utiliser les points d'intérêt du GPX", value=True)
    saisie_ravitos = st.text_input("Ravitos supplémentaires (km)", placeholder="ex. 5.2, 10, 14.5")
    saisie_barrieres = st.text_area(
        "Barrières (km ; temps écoulé max)", placeholder="ex.\n12.5 ; 2:30", height=80
    )

    st.header("Athlète")
    for nom, libelle, aide in [
        ("derive_fatigue", "Dérive de fatigue", "0.10 = 10 % plus lent en fin de course."),
        ("facteur_descente_min", "Plancher de descente",
         "0.8 = jamais plus de 1.25x plus vite qu'à plat en descente."),
        ("glucides_g_h", "Glucides (g/h)", None),
        ("intervalle_min", "Intervalle entre prises (min)", None),
        ("delai_premiere_prise_min", "Pas de prise avant (min)", None),
        ("temperature_c", "Température prévue (°C)", "Pilote la quantité d'eau."),
    ]:
        mini, maxi, pas = CURSEURS[nom]
        st.slider(libelle, mini, maxi, step=pas, key=nom, help=aide)

    st.header("Réglages d'analyse")
    seuil_modere = st.slider("Seuil « modéré »", 0.05, 0.60,
                             float(CFG.difficulte.seuil_modere), step=0.01,
                             help="Score de difficulté à partir duquel une zone est modérée.")
    seuil_dur = st.slider("Seuil « dur »", 0.10, 0.90, float(CFG.difficulte.seuil_dur),
                          step=0.01, help="Score à partir duquel une zone est dure.")
    lissage_m = st.slider("Lissage du score (m)", 0, 1000,
                          int(CFG.difficulte.lissage_fenetre_m), step=50,
                          help="Moyenne glissante : plus large = zones plus stables.")
    marge_min = st.slider("Marge autour des prises (± min)", 0, 15,
                          int(CFG.decalage.fenetre_min), step=1,
                          help="Décalage autorisé et largeur maximale de la fenêtre de prise.")

    st.subheader("Profil")
    st.download_button("Enregistrer le profil (JSON)", lire_profil().vers_json(),
                       file_name="profil_athlete.json", mime="application/json")
    st.file_uploader("Charger un profil", type=["json"], key="fichier_profil",
                     on_change=charger_profil_json)
    if "message_profil" in st.session_state:
        niveau, texte = st.session_state.pop("message_profil")
        (st.success if niveau == "ok" else st.error)(texte)


# --- Calcul -------------------------------------------------------------------------------

st.title("Plan de ravitaillement trail")
st.caption(
    "Outil d'aide à la planification : ce n'est ni un conseil médical, ni un conseil diététique."
)

cfg = lire_profil().appliquer(CFG)
cfg = replace(
    cfg,
    difficulte=replace(cfg.difficulte, seuil_modere=seuil_modere, seuil_dur=seuil_dur,
                       lissage_fenetre_m=lissage_m),
    decalage=replace(cfg.decalage, fenetre_min=marge_min),
)
try:
    valider(cfg)
except ConfigError as erreur:
    st.error(f"Réglages d'analyse incohérents. {erreur}")
    st.stop()
temps_cible_s = heures * 3600 + minutes * 60

try:
    contenu = fichier.getvalue() if fichier else GPX_EXEMPLE.read_bytes()
    trace, pois = analyser_gpx(contenu)
    ravitos, barrieres = rassembler_points(
        cfg,
        pois if utiliser_pois else None,
        heure_depart,
        parser_ravitos(saisie_ravitos),
        parser_barrieres(saisie_barrieres),
    )
    res = planifier(trace, cfg, temps_cible_s, ravitos, barrieres, heure_depart, decaler)
except (GpxError, PacingError, NutritionError, ValueError) as erreur:
    st.error(str(erreur))
    st.stop()

d_plus, d_moins = denivele(trace["altitude_m"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Distance", f"{trace['distance_m'].iloc[-1] / 1000:.1f} km")
c2.metric("D+ / D-", f"{d_plus:.0f} / {d_moins:.0f} m")
c3.metric("Allure à plat équivalente", f"{formater_mmss(allure_plate(res.segments))}/km")
c4.metric("Glucides au total", f"{res.plan['glucides_g'].sum():.0f} g")
c5.metric("Eau au total", f"{res.plan['eau_ml'].sum() / 1000:.1f} L" if len(res.plan) else "0 L")

if not res.barrieres.empty and not res.barrieres["ok"].all():
    st.error("⚠ Au moins une barrière horaire n'est pas tenue avec ce temps cible.")

onglets = st.tabs(
    ["Plan", "Allure & passages", "Difficulté", "Points d'intérêt", "Comparer", "Calibration"]
)

# --- Onglet Plan --------------------------------------------------------------------------
with onglets[0]:
    st.plotly_chart(
        profil_altimetrique(trace, res.plan, res.segments, res.ravitos),
        use_container_width=True,
    )
    st.subheader("Prises")
    st.caption("Chaque prise a une fenêtre : plus étroite quand c'est dur, jamais en descente "
               "technique ni en pleine montée dure. Le km idéal est indiqué en complément.")
    st.dataframe(
        res.plan.assign(niveau=res.plan["niveau"].map(ETIQUETTES_NIVEAU))[
            ["fenetre", "heure_min", "heure_max", "glucides_g", "eau_ml", "sodium_mg", "niveau",
             "km", "d_plus_cumul_m", "raison"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "fenetre": "Prise recommandée",
            "heure_min": "Entre",
            "heure_max": "et",
            "niveau": "Terrain",
            "km": st.column_config.NumberColumn("Km idéal", format="%.2f"),
            "d_plus_cumul_m": st.column_config.NumberColumn("D+ cumulé (m)", format="%.0f"),
            "glucides_g": st.column_config.NumberColumn("Glucides (g)", format="%.0f"),
            "eau_ml": st.column_config.NumberColumn("Eau (ml)", format="%.0f"),
            "sodium_mg": st.column_config.NumberColumn("Sodium (mg)", format="%.0f"),
            "raison": "Raison du placement",
        },
    )
    st.download_button("Télécharger le plan (CSV)", exporter_csv(res.plan),
                       file_name="plan_ravitaillement.csv", mime="text/csv")

    gauche, droite = st.columns(2)
    with gauche:
        st.subheader("Total à prévoir")
        st.dataframe(total_a_prevoir(res.plan), hide_index=True, use_container_width=True)
    with droite:
        st.subheader("Barrières horaires")
        if res.barrieres.empty:
            st.caption("Aucune barrière (GPX, config ou saisie).")
        else:
            controle = res.barrieres.copy()
            controle["statut"] = controle["ok"].map({True: "✅ tenue", False: "❌ ratée"})
            st.dataframe(controle.drop(columns="ok"), hide_index=True, use_container_width=True)

# --- Onglet Allure ------------------------------------------------------------------------
with onglets[1]:
    st.plotly_chart(graphique_allure(res.segments), use_container_width=True)
    st.subheader("Temps de passage")
    st.dataframe(passages_par_km(res.segments, heure_depart), hide_index=True,
                 use_container_width=True)

# --- Onglet Difficulté --------------------------------------------------------------------
with onglets[2]:
    st.write(f"**Zones** : facile < {cfg.difficulte.seuil_modere:.2f} ≤ modéré < "
             f"{cfg.difficulte.seuil_dur:.2f} ≤ dur (score continu de 0 à 1, lissé)")
    st.dataframe(
        res.zones.assign(niveau=res.zones["niveau"].map(ETIQUETTES_NIVEAU)),
        hide_index=True,
        use_container_width=True,
        column_config={
            "km_debut": st.column_config.NumberColumn("Du km", format="%.1f"),
            "km_fin": st.column_config.NumberColumn("Au km", format="%.1f"),
            "niveau": "Niveau",
            "score_moyen": st.column_config.NumberColumn("Score moyen", format="%.2f"),
            "score_max": st.column_config.NumberColumn("Score max", format="%.2f"),
        },
    )
    st.caption("Détail par segment :")
    colonnes = ["debut_m", "fin_m", "pente", "mode", "allure_s_km", "difficulte", "niveau",
                "f_montee", "f_fatigue", "f_position", "f_descente", "f_enchainement", "f_creux"]
    st.dataframe(res.segments[colonnes].round(3), hide_index=True, use_container_width=True)

# --- Onglet Points d'intérêt --------------------------------------------------------------
with onglets[3]:
    if pois.empty:
        st.caption("Le GPX ne contient aucun point d'intérêt (<wpt>).")
    else:
        affichage = pois.assign(categories=pois["categories"].map(", ".join))
        st.dataframe(affichage, hide_index=True, use_container_width=True,
                     column_config={"km": st.column_config.NumberColumn("Km", format="%.2f"),
                                    "ecart_trace_m": st.column_config.NumberColumn(
                                        "Écart à la trace (m)", format="%.0f")})
        hors = pois["km"].isna().sum()
        if hors:
            st.caption(f"{hors} point(s) ignoré(s) : à plus de "
                       f"{CFG.waypoints.distance_max_trace_m:.0f} m de la trace.")
    st.write("**Ravitos retenus** : " + (", ".join(
        f"{r.nom} (km {r.km:.1f}, {r.type})" for r in res.ravitos) or "aucun"))

# --- Onglet Comparer ----------------------------------------------------------------------
with onglets[4]:
    ch, cm = st.columns(2)
    heures_b = ch.number_input("Scénario B (h)", 0, 72, value=int(heures), key="h_b")
    minutes_b = cm.number_input("(min)", 0, 59, value=min(int(minutes) + 15, 59), key="m_b")
    try:
        res_b = planifier(trace, cfg, heures_b * 3600 + minutes_b * 60, ravitos, barrieres,
                          heure_depart, decaler)
    except (PacingError, NutritionError) as erreur:
        st.error(f"Scénario B : {erreur}")
    else:
        def resume(r: Resultat, temps_s: float) -> dict[str, str]:
            lignes = {
                "Temps cible": formater_hmm(temps_s),
                "Allure à plat": f"{formater_mmss(allure_plate(r.segments))}/km",
                "Prises": str(len(r.plan)),
                "Glucides": f"{r.plan['glucides_g'].sum():.0f} g",
                "Eau": f"{r.plan['eau_ml'].sum() / 1000:.1f} L" if len(r.plan) else "0 L",
            }
            for ravito in r.ravitos:
                lignes[f"Passage {ravito.nom}"] = formater_hmm(float(temps_au_km(r.segments,
                                                                                 ravito.km)))
            for _, b in r.barrieres.iterrows():
                lignes[f"{b['nom']} (limite {b['limite']})"] = (
                    f"{b['passage_prevu']} ({b['marge_min']:+d} min)")
            return lignes

        st.dataframe(
            pd.DataFrame({"A": resume(res, temps_cible_s),
                          "B": resume(res_b, heures_b * 3600 + minutes_b * 60)}),
            use_container_width=True,
        )

# --- Onglet Calibration -------------------------------------------------------------------
with onglets[5]:
    st.write(
        "Importe des **activités passées** (GPX horodatés exportés de ta montre) pour ajuster "
        "la dérive de fatigue et le plancher de descente à ta façon de courir."
    )
    activites = st.file_uploader("Activités GPX", type=["gpx"], accept_multiple_files=True,
                                 key="activites")
    if activites:
        segments_activites = []
        for f in activites:
            try:
                trace_a, _ = analyser_gpx(f.getvalue())
            except GpxError as erreur:
                st.warning(f"{f.name} : {erreur}")
                continue
            if "temps_reel_s" not in trace_a:
                st.warning(f"{f.name} : pas d'horodatage, ignorée.")
                continue
            segments_activites.append(decouper_segments(trace_a, CFG.segments))
        try:
            calib = calibrer(segments_activites, cfg.pacing, CFG.calibration)
        except CalibrationError as erreur:
            st.error(str(erreur))
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Dérive de fatigue", f"{calib.derive_fatigue:.2f}",
                      f"{calib.derive_fatigue - cfg.pacing.derive_fatigue:+.2f}")
            k2.metric("Plancher de descente", f"{calib.facteur_descente_min:.2f}",
                      f"{calib.facteur_descente_min - cfg.pacing.facteur_descente_min:+.2f}")
            k3.metric("Allure à plat mesurée", f"{formater_mmss(calib.allure_plate_s_km)}/km")
            st.caption(f"Calculé sur {calib.nb_activites} activité(s) exploitable(s).")
            st.button("Appliquer à mon profil", on_click=appliquer_calibration,
                      args=(calib.derive_fatigue, calib.facteur_descente_min))
